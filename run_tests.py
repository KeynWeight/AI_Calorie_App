#!/usr/bin/env python3
"""
Test runner script for CI/CD pipeline.
Provides different test configurations for various scenarios.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'=' * 50}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'=' * 50}")

    # Prefix python commands with uv run
    if cmd[0] == "python":
        cmd = ["uv", "run"] + cmd

    try:
        subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"{description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description or 'Command'} failed with exit code {e.returncode}")
        return False


def run_unit_tests():
    """Run fast unit tests only (for CI/CD)."""
    cmd = ["python", "-m", "pytest", "-m", "unit", "--tb=short", "-v", "--durations=10"]
    return run_command(cmd, "Unit Tests (CI/CD)")


def run_all_tests():
    """Run all tests including slower ones."""
    cmd = ["python", "-m", "pytest", "-v", "--tb=short", "--durations=10"]
    return run_command(cmd, "All Tests")


def run_specific_category(category):
    """Run tests for a specific category."""
    cmd = ["python", "-m", "pytest", "-m", category, "-v", "--tb=short"]
    return run_command(cmd, f"Tests for category: {category}")


def run_coverage_tests():
    """Run tests with coverage reporting."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "-m",
        "unit",
        "--cov=calorie_app",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--tb=short",
        "-v",
    ]
    return run_command(cmd, "Unit Tests with Coverage")


def run_linting():
    """Run code linting and formatting checks."""
    commands = [(["python", "-m", "ruff", "check", "."], "Ruff Linting & Import Check")]

    all_passed = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            all_passed = False

    return all_passed


def run_security_checks():
    """Run security checks."""
    commands = [
        (["python", "-m", "bandit", "-r", "src/"], "Security Scan (Bandit)"),
        # Note: Safety check temporarily disabled due to interactive login requirement
        # (["python", "-m", "safety", "scan", "--disable-optional-telemetry-data"], "Dependency Security Check"),
    ]

    all_passed = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            all_passed = False

    return all_passed


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test runner for Calorie App")
    parser.add_argument(
        "--ci", action="store_true", help="Run CI/CD tests (fast unit tests only)"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage"
    )
    parser.add_argument(
        "--lint", action="store_true", help="Run linting and formatting checks"
    )
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument(
        "--category",
        help="Run tests for specific category (unit, nutrition, vision, etc.)",
    )
    parser.add_argument(
        "--full-pipeline", action="store_true", help="Run complete CI/CD pipeline"
    )

    args = parser.parse_args()

    success = True

    if args.ci or (not any(vars(args).values())):  # Default to CI mode
        print("Running CI/CD Pipeline Tests")
        success = run_unit_tests()

    elif args.all:
        print("Running All Tests")
        success = run_all_tests()

    elif args.coverage:
        print("Running Tests with Coverage")
        success = run_coverage_tests()

    elif args.lint:
        print("Running Code Quality Checks")
        success = run_linting()

    elif args.security:
        print("Running Security Checks")
        success = run_security_checks()

    elif args.category:
        print(f"Running {args.category.title()} Tests")
        success = run_specific_category(args.category)

    elif args.full_pipeline:
        print("Running Full CI/CD Pipeline")
        steps = [
            ("Unit Tests", run_unit_tests),
            ("Unit Tests with Coverage", run_coverage_tests),
            ("Code Quality", run_linting),
            ("Security Checks", run_security_checks),
        ]

        for step_name, step_func in steps:
            print(f"\nRunning {step_name}...")
            if not step_func():
                print(f"{step_name} failed!")
                success = False
                break
            print(f"{step_name} passed!")

    if success:
        print("\nAll tests passed! Ready for deployment.")
        sys.exit(0)
    else:
        print("\nSome tests failed! Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
