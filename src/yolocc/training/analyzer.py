#!/usr/bin/env python3
"""
Training Analyzer - Active Learning Module
==========================================

Analyzes trained models to identify problematic images for re-annotation.

Features:
1. Identifies uncertain predictions (confidence in specified range)
2. Detects potential false negatives (no detection but has labels)
3. Generates reports for prioritized re-annotation

Classes:
    TrainingAnalyzer: Main class for active learning analysis

Functions:
    analyze_training: Convenience function for TrainingAnalyzer
    analyze_cli: CLI entry point
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any

from ultralytics import YOLO
import yaml

from yolocc.paths import get_reports_root, resolve_workspace_path


class TrainingAnalyzer:
    """Analyzes a trained model for active learning."""

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        low_conf_threshold: float = 0.4,
        high_conf_threshold: float = 0.65,
        output_dir: Optional[str] = "reports"
    ):
        """
        Initialize the analyzer.

        Args:
            model_path: Path to the trained model (.pt)
            dataset_path: Path to the dataset directory
            low_conf_threshold: Low confidence threshold (below = very uncertain)
            high_conf_threshold: High confidence threshold (above = confident)
            output_dir: Directory for reports (None to disable saving)

        Raises:
            ValueError: If thresholds are invalid
        """
        # Validate thresholds
        if not (0 <= low_conf_threshold <= 1):
            raise ValueError(f"low_conf_threshold must be between 0 and 1, got {low_conf_threshold}")
        if not (0 <= high_conf_threshold <= 1):
            raise ValueError(f"high_conf_threshold must be between 0 and 1, got {high_conf_threshold}")
        if low_conf_threshold >= high_conf_threshold:
            raise ValueError(
                f"low_conf_threshold ({low_conf_threshold}) must be less than "
                f"high_conf_threshold ({high_conf_threshold})"
            )

        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.low_conf = low_conf_threshold
        self.high_conf = high_conf_threshold
        self.medium_high_boundary = self.high_conf + (1.0 - self.high_conf) / 2
        self.output_dir = Path(output_dir) if output_dir else None

        # Results storage
        self.results: Dict[str, Any] = {
            'total_images': 0,
            'total_detections': 0,
            'avg_confidence': 0.0,
            'confidence_buckets': defaultdict(int),
            'uncertain_images': [],
            'no_detection_images': [],
            'false_negative_candidates': [],
        }

        # Model (loaded lazily)
        self.model: Optional[YOLO] = None
        self.data_yaml: Optional[Dict] = None
        self._all_confidences: List[float] = []

    def analyze(self) -> Dict:
        """
        Execute complete analysis.

        Returns:
            Dictionary with analysis results
        """
        print("=" * 60)
        print("Training Analysis (Active Learning)")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Confidence thresholds: {self.low_conf} - {self.high_conf}")
        print()

        # Load model
        print("Loading model...")
        self.model = YOLO(str(self.model_path))

        # Load data.yaml for labels
        data_yaml_path = self.dataset_path / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path) as f:
                self.data_yaml = yaml.safe_load(f)
        else:
            print(f"Warning: data.yaml not found at {data_yaml_path}")
            self.data_yaml = None

        # Analyze each split - support both folder structures:
        # Structure 1: images/train, labels/train (YOLO default)
        # Structure 2: train/images, train/labels (Roboflow export)
        split_configs = [
            ('train', 'images/train', 'labels/train'),
            ('train', 'train/images', 'train/labels'),
            ('val', 'images/val', 'labels/val'),
            ('val', 'valid/images', 'valid/labels'),
        ]

        analyzed_splits = set()
        for split, img_rel, lbl_rel in split_configs:
            if split in analyzed_splits:
                continue

            img_dir = self.dataset_path / img_rel
            lbl_dir = self.dataset_path / lbl_rel

            if not img_dir.exists():
                continue

            analyzed_splits.add(split)
            print(f"\nAnalyzing {split} split ({img_rel})...")
            self._analyze_split(img_dir, lbl_dir, split)

        # Compute global avg confidence across all splits
        if self._all_confidences:
            self.results['avg_confidence'] = sum(self._all_confidences) / len(self._all_confidences)

        # Generate report
        self._generate_report()

        return self.results

    def _analyze_split(self, img_dir: Path, lbl_dir: Path, split: str) -> None:
        """Analyze a dataset split."""
        images = list(img_dir.glob('*'))
        images = [p for p in images if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]

        all_confidences: List[float] = []

        for i, img_path in enumerate(images):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(images)} images...")

            self.results['total_images'] += 1

            # Inference with low confidence to see all detections
            results = self.model.predict(str(img_path), verbose=False, conf=0.1)

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                # No detections
                self.results['no_detection_images'].append({
                    'path': str(img_path),
                    'split': split
                })

                # Check if it has labels (potential false negative)
                label_path = lbl_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    content = label_path.read_text().strip()
                    if content:  # Non-empty labels
                        num_labels = len(content.split('\n'))
                        self.results['false_negative_candidates'].append({
                            'path': str(img_path),
                            'split': split,
                            'num_labels': num_labels
                        })
                continue

            # Analyze detections
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()

            self.results['total_detections'] += len(confidences)
            all_confidences.extend(confidences.tolist())
            self._all_confidences.extend(confidences.tolist())

            # Classify by confidence bucket (using configurable thresholds)
            for conf in confidences:
                if conf < self.low_conf:
                    self.results['confidence_buckets']['very_low'] += 1
                elif conf < self.high_conf:
                    self.results['confidence_buckets']['uncertain'] += 1
                elif conf < self.medium_high_boundary:
                    self.results['confidence_buckets']['medium'] += 1
                else:
                    self.results['confidence_buckets']['high'] += 1

            # Check for uncertain detections
            uncertain_detections = [c for c in confidences if self.low_conf <= c <= self.high_conf]
            if uncertain_detections:
                self.results['uncertain_images'].append({
                    'path': str(img_path),
                    'split': split,
                    'num_uncertain': len(uncertain_detections),
                    'avg_conf': float(sum(uncertain_detections) / len(uncertain_detections)),
                    'min_conf': float(min(uncertain_detections))
                })

        print(f"  Done: {len(images)} images analyzed")

    def _generate_report(self) -> None:
        """Generate analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print()
        print("=" * 60)
        print("Analysis Report")
        print("=" * 60)

        # General stats
        print("\nGeneral Statistics:")
        print(f"   Total images analyzed: {self.results['total_images']}")
        print(f"   Total detections: {self.results['total_detections']}")
        print(f"   Average confidence: {self.results['avg_confidence']:.3f}")

        # Confidence distribution
        buckets = self.results['confidence_buckets']
        total_det = self.results['total_detections'] or 1
        print("\nConfidence Distribution:")
        print(f"   Very low (<{self.low_conf}):  {buckets['very_low']} ({buckets['very_low']/total_det*100:.1f}%)")
        print(f"   Uncertain ({self.low_conf}-{self.high_conf}): {buckets['uncertain']} ({buckets['uncertain']/total_det*100:.1f}%)")
        print(f"   Medium ({self.high_conf}-{self.medium_high_boundary}): {buckets['medium']} ({buckets['medium']/total_det*100:.1f}%)")
        print(f"   High (>{self.medium_high_boundary}):      {buckets['high']} ({buckets['high']/total_det*100:.1f}%)")

        # Priority images
        print("\nPriority Images for Review:")

        fn_candidates = self.results['false_negative_candidates']
        print("\n   HIGH PRIORITY - False Negative Candidates:")
        print("   (No detection but has labels - likely missed objects)")
        print(f"   Count: {len(fn_candidates)}")
        if fn_candidates[:5]:
            for item in fn_candidates[:5]:
                print(f"      -> {item['path']} ({item['num_labels']} labels)")
            if len(fn_candidates) > 5:
                print(f"      ... and {len(fn_candidates) - 5} more")

        uncertain = self.results['uncertain_images']
        print("\n   MEDIUM PRIORITY - Uncertain Images:")
        print(f"   (Detections with confidence {self.low_conf}-{self.high_conf})")
        print(f"   Count: {len(uncertain)}")
        uncertain_sorted = sorted(uncertain, key=lambda x: x['min_conf'])
        if uncertain_sorted[:5]:
            for item in uncertain_sorted[:5]:
                print(f"      -> {item['path']} (min_conf: {item['min_conf']:.3f})")
            if len(uncertain_sorted) > 5:
                print(f"      ... and {len(uncertain_sorted) - 5} more")

        no_det = self.results['no_detection_images']
        print("\n   LOW PRIORITY - No Detection (no labels):")
        print("   (May be background images or very difficult cases)")
        fn_paths = [y['path'] for y in fn_candidates]
        no_det_no_label = [x for x in no_det if x['path'] not in fn_paths]
        print(f"   Count: {len(no_det_no_label)}")

        # Save to file if output_dir is set
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            uncertain_path = self.output_dir / "uncertain_images.txt"

            with open(uncertain_path, 'w') as f:
                f.write("# Uncertain Images for Re-annotation\n")
                f.write(f"# Generated: {timestamp}\n")
                f.write(f"# Model: {self.model_path}\n")
                f.write("#\n")
                f.write("# HIGH PRIORITY - False Negative Candidates\n")
                for item in fn_candidates:
                    f.write(f"{item['path']}\n")
                f.write("#\n")
                f.write("# MEDIUM PRIORITY - Uncertain Detections\n")
                for item in uncertain_sorted:
                    f.write(f"{item['path']}\n")

            print("\nFiles saved:")
            print(f"   -> {uncertain_path}")

        # Recommendations
        print("\nRecommendations:")

        if len(fn_candidates) > 0:
            print(f"   1. Review {len(fn_candidates)} false negative candidates first")
            print("      These images have labels but no detections - check if:")
            print("      - Annotations are correct")
            print("      - Objects are too small/occluded")
            print("      - Need more similar training data")

        if len(uncertain) > 50:
            print(f"   2. {len(uncertain)} images have uncertain detections")
            print("      Consider re-annotating the top uncertain images")
            print("      Push to CVAT for review: yolo-cvat push --from-analysis reports/uncertain_images.txt")

        if self.results['avg_confidence'] < 0.6:
            print(f"   3. Average confidence is low ({self.results['avg_confidence']:.3f})")
            print("      Consider: more training epochs, more data, or check annotation quality")


def analyze_training(
    model_path: str,
    dataset_path: str,
    low_conf: float = 0.4,
    high_conf: float = 0.65,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Analyze a trained model for active learning.

    Args:
        model_path: Path to the trained model (.pt)
        dataset_path: Path to the dataset directory
        low_conf: Low confidence threshold
        high_conf: High confidence threshold
        output_dir: Directory for reports (None = use default workspace reports directory)

    Returns:
        Dictionary with analysis results
    """
    resolved_output = output_dir if output_dir is not None else str(get_reports_root())

    analyzer = TrainingAnalyzer(
        model_path=model_path,
        dataset_path=dataset_path,
        low_conf_threshold=low_conf,
        high_conf_threshold=high_conf,
        output_dir=resolved_output
    )
    return analyzer.analyze()


def analyze_cli() -> None:
    """CLI entry point for training analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze trained model for active learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yolo-analyze --model models/my_model.pt --dataset datasets/my_data
  yolo-analyze --model models/variant_v1.pt --dataset datasets/variant --low-conf 0.5
        """
    )
    parser.add_argument("--model", "-m", required=True,
                        help="Path to trained model (.pt) (relative paths are resolved from workspace root)")
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to dataset directory (relative paths are resolved from workspace root)")
    parser.add_argument("--low-conf", type=float, default=0.4, help="Low confidence threshold (default: 0.4)")
    parser.add_argument("--high-conf", type=float, default=0.65, help="High confidence threshold (default: 0.65)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for reports (default: workspace reports directory)")
    parser.add_argument("--upload-cvat", action="store_true",
                        help="Push uncertain images to CVAT for review after analysis")
    args = parser.parse_args()

    try:
        model_path = resolve_workspace_path(args.model)
        dataset_path = resolve_workspace_path(args.dataset)
        output_dir = resolve_workspace_path(args.output) if args.output else None

        analyze_training(
            model_path=str(model_path),
            dataset_path=str(dataset_path),
            low_conf=args.low_conf,
            high_conf=args.high_conf,
            output_dir=str(output_dir) if output_dir is not None else None
        )

        if args.upload_cvat:
            from yolocc.cvat.push import push_from_analysis
            report_dir = str(output_dir) if output_dir is not None else str(get_reports_root())
            analysis_file = str(Path(report_dir) / "uncertain_images.txt")
            print("\nUploading uncertain images to CVAT...")
            push_from_analysis(analysis_file)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


__all__ = [
    "TrainingAnalyzer",
    "analyze_training",
    "analyze_cli",
]


if __name__ == "__main__":
    analyze_cli()
