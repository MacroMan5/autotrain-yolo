#!/usr/bin/env python3
"""
Annotation Merger
=================

Merge and remap YOLO annotations from multiple sources.

Features:
- Merge annotations from multiple directories
- Remap class IDs (e.g., source class 7 -> target class 1)
- Filter to keep only specific classes
- Smart deduplication with configurable conflict resolution

Classes:
    BBox: YOLO bounding box representation

Functions:
    merge_annotations: Main function to merge annotations
    deduplicate_bboxes: Remove duplicate bounding boxes
    load_annotations: Load annotations from file
    save_annotations: Save annotations to file
    merger_cli: CLI entry point
"""

import argparse
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from yolocc.paths import resolve_workspace_path


@dataclass
class BBox:
    """YOLO bounding box representation with source tracking."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    source_idx: int = 0  # Track which source this came from

    def to_yolo_line(self) -> str:
        """Convert to YOLO format line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str, source_idx: int = 0) -> "BBox":
        """Parse from YOLO format line."""
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO line: {line}")
        return cls(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
            source_idx=source_idx
        )

    @property
    def area(self) -> float:
        """Calculate box area (normalized)."""
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        """Calculate IoU with another bounding box."""
        # Convert to corner format
        x1_a = self.x_center - self.width / 2
        y1_a = self.y_center - self.height / 2
        x2_a = self.x_center + self.width / 2
        y2_a = self.y_center + self.height / 2

        x1_b = other.x_center - other.width / 2
        y1_b = other.y_center - other.height / 2
        x2_b = other.x_center + other.width / 2
        y2_b = other.y_center + other.height / 2

        # Intersection
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area_a = self.width * self.height
        area_b = other.width * other.height
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0


def parse_remap(remap_args: List[str]) -> Dict[int, int]:
    """Parse remap arguments like '7:1' into {7: 1}."""
    remap = {}
    for arg in remap_args:
        if ':' not in arg:
            raise ValueError(f"Invalid remap format '{arg}'. Use 'source:target' (e.g., '7:1')")
        source, target = arg.split(':')
        remap[int(source)] = int(target)
    return remap


def load_annotations(label_path: Path, source_idx: int = 0) -> List[BBox]:
    """Load YOLO annotations from a label file."""
    if not label_path.exists():
        return []

    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    bboxes.append(BBox.from_yolo_line(line, source_idx))
                except ValueError as e:
                    print(f"Warning: Skipping invalid line in {label_path}: {e}")
    return bboxes


def save_annotations(label_path: Path, bboxes: List[BBox]) -> None:
    """Save YOLO annotations to a label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            f.write(bbox.to_yolo_line() + '\n')


def choose_best_bbox(
    bbox_a: BBox,
    bbox_b: BBox,
    prefer_smaller: bool = False,
    source_priority: str = "first"
) -> BBox:
    """
    Choose the best bbox when two overlap.

    Args:
        bbox_a: First bounding box
        bbox_b: Second bounding box
        prefer_smaller: If True, prefer the smaller (tighter) box
        source_priority: "first" = prefer lower source_idx, "last" = prefer higher

    Returns:
        The chosen bounding box
    """
    if prefer_smaller:
        # Prefer smaller/tighter box (more precise)
        if bbox_a.area < bbox_b.area:
            return bbox_a
        elif bbox_b.area < bbox_a.area:
            return bbox_b
        # If equal area, fall through to source priority

    # Apply source priority
    if source_priority == "first":
        return bbox_a if bbox_a.source_idx <= bbox_b.source_idx else bbox_b
    else:  # "last"
        return bbox_a if bbox_a.source_idx >= bbox_b.source_idx else bbox_b


def deduplicate_bboxes(
    bboxes: List[BBox],
    iou_threshold: float = 0.5,
    prefer_smaller: bool = False,
    source_priority: str = "first"
) -> Tuple[List[BBox], Dict]:
    """
    Remove duplicate bounding boxes based on IoU threshold.

    Args:
        bboxes: List of bounding boxes
        iou_threshold: IoU threshold for considering duplicates
        prefer_smaller: Prefer smaller boxes in conflicts
        source_priority: "first" or "last" source preference

    Returns:
        Tuple of (deduplicated boxes, conflict stats)
    """
    if not bboxes:
        return [], {}

    conflict_stats = {
        'total_conflicts': 0,
        'kept_smaller': 0,
        'kept_larger': 0,
        'kept_first_source': 0,
        'kept_last_source': 0
    }

    # Group by class
    by_class = defaultdict(list)
    for bbox in bboxes:
        by_class[bbox.class_id].append(bbox)

    result = []
    for class_id, class_bboxes in by_class.items():
        keep = []
        for bbox in class_bboxes:
            is_duplicate = False
            for i, kept in enumerate(keep):
                if bbox.iou(kept) > iou_threshold:
                    is_duplicate = True
                    conflict_stats['total_conflicts'] += 1

                    # Choose the best one
                    winner = choose_best_bbox(kept, bbox, prefer_smaller, source_priority)

                    # Track stats
                    if prefer_smaller:
                        if winner.area <= min(kept.area, bbox.area):
                            conflict_stats['kept_smaller'] += 1
                        else:
                            conflict_stats['kept_larger'] += 1

                    if winner.source_idx == kept.source_idx:
                        if source_priority == "first":
                            conflict_stats['kept_first_source'] += 1
                        else:
                            conflict_stats['kept_last_source'] += 1
                    else:
                        if source_priority == "first":
                            conflict_stats['kept_last_source'] += 1
                        else:
                            conflict_stats['kept_first_source'] += 1

                    # Replace if new one is better
                    if winner is bbox:
                        keep[i] = bbox
                    break

            if not is_duplicate:
                keep.append(bbox)
        result.extend(keep)

    return result, conflict_stats


def merge_annotations(
    sources: List[Path],
    output: Path,
    remap: Optional[Dict[int, int]] = None,
    keep_classes: Optional[Set[int]] = None,
    iou_threshold: float = 0.5,
    prefer_smaller: bool = False,
    source_priority: str = "first",
    dry_run: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Merge annotations from multiple sources.

    Args:
        sources: List of source label directories
        output: Output directory for merged labels
        remap: Dict mapping source class IDs to target class IDs
        keep_classes: Set of class IDs to keep (after remapping). None = keep all
        iou_threshold: IoU threshold for deduplication
        prefer_smaller: If True, prefer smaller boxes in conflicts
        source_priority: "first" or "last" - which source to prefer
        dry_run: If True, don't write files, just report what would happen
        verbose: Print detailed progress

    Returns:
        Statistics dict with counts
    """
    remap = remap or {}
    stats = {
        'total_images': 0,
        'images_with_annotations': 0,
        'total_bboxes_input': 0,
        'total_bboxes_output': 0,
        'bboxes_remapped': 0,
        'bboxes_filtered': 0,
        'bboxes_deduplicated': 0,
        'conflicts': {
            'total': 0,
            'kept_smaller': 0,
            'kept_larger': 0,
            'kept_first_source': 0,
            'kept_last_source': 0
        },
        'source_stats': {str(s): {'files': 0, 'bboxes': 0} for s in sources}
    }

    # Collect all label files from all sources
    all_label_files = set()
    for source in sources:
        source_path = Path(source)
        if not source_path.exists():
            print(f"Warning: Source directory does not exist: {source}")
            continue
        for label_file in source_path.glob('*.txt'):
            all_label_files.add(label_file.stem)

    stats['total_images'] = len(all_label_files)

    if verbose:
        print(f"Found {len(all_label_files)} unique label files across {len(sources)} sources")

    # Process each label file
    for label_stem in sorted(all_label_files):
        merged_bboxes = []

        # Collect bboxes from all sources (with source index)
        for source_idx, source in enumerate(sources):
            source_path = Path(source)
            label_path = source_path / f"{label_stem}.txt"
            bboxes = load_annotations(label_path, source_idx)

            if bboxes:
                stats['source_stats'][str(source)]['files'] += 1
                stats['source_stats'][str(source)]['bboxes'] += len(bboxes)

            stats['total_bboxes_input'] += len(bboxes)
            merged_bboxes.extend(bboxes)

        if not merged_bboxes:
            continue

        stats['images_with_annotations'] += 1

        # Apply remapping
        remapped_bboxes = []
        for bbox in merged_bboxes:
            if bbox.class_id in remap:
                bbox = replace(bbox, class_id=remap[bbox.class_id])
                stats['bboxes_remapped'] += 1
            remapped_bboxes.append(bbox)

        # Filter classes
        if keep_classes is not None:
            filtered_bboxes = [b for b in remapped_bboxes if b.class_id in keep_classes]
            stats['bboxes_filtered'] += len(remapped_bboxes) - len(filtered_bboxes)
            remapped_bboxes = filtered_bboxes

        # Deduplicate with smart conflict resolution
        before_dedup = len(remapped_bboxes)
        final_bboxes, conflict_stats = deduplicate_bboxes(
            remapped_bboxes,
            iou_threshold,
            prefer_smaller,
            source_priority
        )
        stats['bboxes_deduplicated'] += before_dedup - len(final_bboxes)

        # Accumulate conflict stats
        stats['conflicts']['total'] += conflict_stats.get('total_conflicts', 0)
        stats['conflicts']['kept_smaller'] += conflict_stats.get('kept_smaller', 0)
        stats['conflicts']['kept_larger'] += conflict_stats.get('kept_larger', 0)
        stats['conflicts']['kept_first_source'] += conflict_stats.get('kept_first_source', 0)
        stats['conflicts']['kept_last_source'] += conflict_stats.get('kept_last_source', 0)

        stats['total_bboxes_output'] += len(final_bboxes)

        # Save
        if not dry_run and final_bboxes:
            output_path = Path(output) / f"{label_stem}.txt"
            save_annotations(output_path, final_bboxes)

        if verbose and final_bboxes:
            print(f"  {label_stem}: {len(merged_bboxes)} -> {len(final_bboxes)} bboxes")

    return stats


def print_stats(stats: Dict, source_names: List[str]) -> None:
    """Print merge statistics."""
    print("\n" + "=" * 60)
    print("MERGE STATISTICS")
    print("=" * 60)
    print(f"Total images processed:     {stats['total_images']}")
    print(f"Images with annotations:    {stats['images_with_annotations']}")
    print(f"Total bboxes (input):       {stats['total_bboxes_input']}")
    print(f"Total bboxes (output):      {stats['total_bboxes_output']}")
    print(f"Bboxes remapped:            {stats['bboxes_remapped']}")
    print(f"Bboxes filtered:            {stats['bboxes_filtered']}")
    print(f"Bboxes deduplicated:        {stats['bboxes_deduplicated']}")

    print("\n" + "-" * 60)
    print("CONFLICT RESOLUTION")
    print("-" * 60)
    print(f"Total conflicts resolved:   {stats['conflicts']['total']}")
    if stats['conflicts']['total'] > 0:
        print(f"  Kept smaller box:         {stats['conflicts']['kept_smaller']}")
        print(f"  Kept larger box:          {stats['conflicts']['kept_larger']}")
        print(f"  Kept from first source:   {stats['conflicts']['kept_first_source']}")
        print(f"  Kept from last source:    {stats['conflicts']['kept_last_source']}")

    print("\n" + "-" * 60)
    print("PER-SOURCE STATISTICS")
    print("-" * 60)
    for i, (source, source_stats) in enumerate(stats['source_stats'].items()):
        source_label = f"[{i}] {source}"
        print(f"  {source_label}:")
        print(f"      Files with annotations: {source_stats['files']}")
        print(f"      Total bboxes:           {source_stats['bboxes']}")
    print("=" * 60)


def merger_cli() -> None:
    """CLI entry point for annotation merging."""
    parser = argparse.ArgumentParser(
        description="Merge and remap YOLO annotations from multiple sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yolo-merge --sources labels_a labels_b --output labels_merged
  yolo-merge --sources labels_a labels_b --output merged --remap 7:1
  yolo-merge --sources labels_a labels_b --output merged --remap 7:1 --keep-classes 0 1
        """
    )

    parser.add_argument(
        '--sources', '-s',
        nargs='+',
        required=True,
        help='Source label directories to merge (order matters for priority, relative paths are resolved from workspace root)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for merged labels (relative paths are resolved from workspace root)'
    )

    parser.add_argument(
        '--remap', '-r',
        nargs='*',
        default=[],
        help='Class ID remapping in format "source:target" (e.g., 7:1 0:0)'
    )

    parser.add_argument(
        '--keep-classes', '-k',
        nargs='*',
        type=int,
        default=None,
        help='Only keep these class IDs (after remapping). Default: keep all'
    )

    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for deduplication (default: 0.5)'
    )

    parser.add_argument(
        '--prefer-smaller',
        action='store_true',
        help='In conflicts, prefer the smaller (tighter/more precise) box'
    )

    parser.add_argument(
        '--source-priority',
        choices=['first', 'last'],
        default='first',
        help='Which source to prefer in conflicts: "first" or "last" (default: first)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress'
    )

    parser.add_argument(
        '--clear-output',
        action='store_true',
        help='Clear output directory before merging'
    )

    args = parser.parse_args()

    # Parse remapping
    try:
        remap = parse_remap(args.remap)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Convert keep_classes to set
    keep_classes = set(args.keep_classes) if args.keep_classes is not None else None

    # Validate sources (resolve relative paths from workspace)
    sources = [resolve_workspace_path(s) for s in args.sources]
    valid_sources = [s for s in sources if s.exists()]

    if not valid_sources:
        print("Error: No valid source directories found")
        sys.exit(1)

    if len(valid_sources) < len(sources):
        missing = [s for s in sources if not s.exists()]
        print(f"Warning: Missing sources: {missing}")

    # Clear output if requested
    output = resolve_workspace_path(args.output)
    if args.clear_output and output.exists() and not args.dry_run:
        shutil.rmtree(output)
        print(f"Cleared output directory: {output}")

    # Print config
    print("\n" + "=" * 60)
    print("MERGE CONFIGURATION")
    print("=" * 60)
    print("Sources (in priority order):")
    for i, s in enumerate(valid_sources):
        print(f"  [{i}] {s}")
    print(f"Output:          {output}")
    print(f"Remap:           {remap if remap else 'None'}")
    print(f"Keep classes:    {sorted(keep_classes) if keep_classes else 'All'}")
    print(f"IoU threshold:   {args.iou_threshold}")
    print(f"Prefer smaller:  {args.prefer_smaller}")
    print(f"Source priority: {args.source_priority}")
    print(f"Dry run:         {args.dry_run}")
    print("=" * 60)

    # Run merge
    stats = merge_annotations(
        sources=valid_sources,
        output=output,
        remap=remap,
        keep_classes=keep_classes,
        iou_threshold=args.iou_threshold,
        prefer_smaller=args.prefer_smaller,
        source_priority=args.source_priority,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    # Print results
    print_stats(stats, [str(s) for s in valid_sources])

    if args.dry_run:
        print("\n[DRY RUN] No files were written.")
    else:
        print(f"\nMerged annotations saved to: {output}")


__all__ = [
    "BBox",
    "merge_annotations",
    "deduplicate_bboxes",
    "load_annotations",
    "save_annotations",
    "parse_remap",
    "merger_cli",
]


if __name__ == '__main__':
    merger_cli()
