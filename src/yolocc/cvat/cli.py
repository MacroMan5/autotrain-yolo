"""CLI entry point for CVAT integration commands."""
from __future__ import annotations

import argparse
import sys


def cvat_cli():
    """Main entry point for yolo-cvat command."""
    parser = argparse.ArgumentParser(
        description="CVAT integration for yolocc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  pull     Pull CVAT annotations as YOLO dataset
  push     Push images to CVAT for review
  deploy   Generate Nuclio function for CVAT auto-annotation

Examples:
  yolo-cvat pull --task 42
  yolo-cvat push --images uncertain/ --task-name "Review batch 1"
  yolo-cvat deploy --model best.pt --name my_detector

Prerequisites:
  pip install "yolocc[cvat]"
  export CVAT_ACCESS_TOKEN=<your-token>
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Pull
    pull_parser = subparsers.add_parser("pull", help="Pull annotations from CVAT")
    pull_group = pull_parser.add_mutually_exclusive_group(required=True)
    pull_group.add_argument("--task", type=int, help="CVAT task ID")
    pull_group.add_argument("--project", type=int, help="CVAT project ID")
    pull_parser.add_argument("--output", type=str, help="Output directory")

    # Push
    push_parser = subparsers.add_parser("push", help="Push images to CVAT")
    push_parser.add_argument("--images", type=str, help="Images directory")
    push_parser.add_argument("--labels", type=str, help="Labels directory (pre-annotations)")
    push_parser.add_argument("--task-name", type=str, default="Review")
    push_parser.add_argument("--project", type=int, help="CVAT project ID")
    push_parser.add_argument("--from-analysis", type=str, help="Analysis file path")

    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model to CVAT via Nuclio")
    deploy_parser.add_argument("--model", type=str, required=True, help="Model path")
    deploy_parser.add_argument("--name", type=str, help="Function name")
    deploy_parser.add_argument("--output", type=str, help="Output directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "pull":
        from yolocc.cvat.pull import pull_task, pull_project
        if args.task:
            pull_task(args.task, args.output)
        else:
            pull_project(args.project, args.output)

    elif args.command == "push":
        from yolocc.cvat.push import push_task, push_from_analysis
        if args.from_analysis:
            push_from_analysis(args.from_analysis)
        elif args.images:
            push_task(args.images, args.task_name, args.labels, args.project)
        else:
            print("ERROR: Provide --images or --from-analysis")
            sys.exit(1)

    elif args.command == "deploy":
        from yolocc.cvat.nuclio import generate_nuclio_function
        generate_nuclio_function(args.model, args.name, args.output)
