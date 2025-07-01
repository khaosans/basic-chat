#!/usr/bin/env python3
"""
ChromaDB Cleanup Script

This script provides comprehensive cleanup functionality for ChromaDB directories
and can be run independently for maintenance purposes.

Usage:
    python scripts/cleanup_chroma.py [--age HOURS] [--force] [--dry-run]
"""

import os
import sys
import argparse
import logging
import shutil
import glob
import time
from pathlib import Path

# Add parent directory to path to import document_processor
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_chroma_directories(age_hours=None, force=False, dry_run=False):
    """
    Clean up ChromaDB directories with various options.
    
    Args:
        age_hours (int, optional): Only clean directories older than this many hours
        force (bool): Force cleanup even if directories are in use
        dry_run (bool): Show what would be cleaned without actually doing it
    """
    logger.info("Starting ChromaDB cleanup")
    
    if dry_run:
        logger.info("DRY RUN MODE - No files will be deleted")
    
    try:
        # Get all ChromaDB directories
        chroma_dirs = glob.glob("./chroma_db*")
        
        if not chroma_dirs:
            logger.info("No ChromaDB directories found")
            return
        
        logger.info(f"Found {len(chroma_dirs)} ChromaDB directories")
        
        cleaned_count = 0
        total_size = 0
        
        for chroma_dir in chroma_dirs:
            try:
                if not os.path.exists(chroma_dir):
                    continue
                
                # Check age if specified
                if age_hours is not None:
                    dir_age = time.time() - os.path.getmtime(chroma_dir)
                    dir_age_hours = dir_age / 3600
                    
                    if dir_age_hours < age_hours:
                        logger.info(f"Skipping {chroma_dir} (age: {dir_age_hours:.1f}h < {age_hours}h)")
                        continue
                
                # Calculate directory size
                dir_size = 0
                for root, dirs, files in os.walk(chroma_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            dir_size += os.path.getsize(file_path)
                        except (OSError, FileNotFoundError):
                            pass
                
                total_size += dir_size
                
                if dry_run:
                    logger.info(f"Would clean: {chroma_dir} (size: {dir_size / 1024 / 1024:.1f} MB)")
                else:
                    # Try to clean up any active instances first
                    if not force:
                        try:
                            DocumentProcessor.cleanup_all_instances()
                        except Exception as e:
                            logger.warning(f"Failed to cleanup instances: {e}")
                    
                    # Remove directory
                    shutil.rmtree(chroma_dir, ignore_errors=True)
                    logger.info(f"Cleaned: {chroma_dir} (size: {dir_size / 1024 / 1024:.1f} MB)")
                    cleaned_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process directory {chroma_dir}: {e}")
        
        # Summary
        if dry_run:
            logger.info(f"DRY RUN SUMMARY: Would clean {len(chroma_dirs)} directories")
        else:
            logger.info(f"CLEANUP SUMMARY: Cleaned {cleaned_count} directories")
        
        logger.info(f"Total size processed: {total_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False
    
    return True

def show_chroma_status():
    """Show current status of ChromaDB directories"""
    logger.info("ChromaDB Status Report")
    logger.info("=" * 50)
    
    try:
        chroma_dirs = glob.glob("./chroma_db*")
        
        if not chroma_dirs:
            logger.info("No ChromaDB directories found")
            return
        
        total_size = 0
        total_files = 0
        
        for chroma_dir in chroma_dirs:
            try:
                if not os.path.exists(chroma_dir):
                    continue
                
                dir_size = 0
                file_count = 0
                
                for root, dirs, files in os.walk(chroma_dir):
                    file_count += len(files)
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            dir_size += os.path.getsize(file_path)
                        except (OSError, FileNotFoundError):
                            pass
                
                dir_age = time.time() - os.path.getmtime(chroma_dir)
                dir_age_hours = dir_age / 3600
                
                logger.info(f"Directory: {chroma_dir}")
                logger.info(f"  Size: {dir_size / 1024 / 1024:.1f} MB")
                logger.info(f"  Files: {file_count}")
                logger.info(f"  Age: {dir_age_hours:.1f} hours")
                logger.info("")
                
                total_size += dir_size
                total_files += file_count
                
            except Exception as e:
                logger.error(f"Error processing {chroma_dir}: {e}")
        
        logger.info(f"TOTAL: {len(chroma_dirs)} directories, {total_files} files, {total_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Clean up ChromaDB directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup_chroma.py                    # Clean all directories
  python scripts/cleanup_chroma.py --age 24           # Clean directories older than 24 hours
  python scripts/cleanup_chroma.py --dry-run          # Show what would be cleaned
  python scripts/cleanup_chroma.py --status           # Show current status
  python scripts/cleanup_chroma.py --force            # Force cleanup even if in use
        """
    )
    
    parser.add_argument(
        '--age', 
        type=int, 
        help='Only clean directories older than AGE hours'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force cleanup even if directories are in use'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be cleaned without actually doing it'
    )
    parser.add_argument(
        '--status', 
        action='store_true', 
        help='Show current status of ChromaDB directories'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.status:
        show_chroma_status()
        return
    
    success = cleanup_chroma_directories(
        age_hours=args.age,
        force=args.force,
        dry_run=args.dry_run
    )
    
    if success:
        logger.info("Cleanup completed successfully")
        sys.exit(0)
    else:
        logger.error("Cleanup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 
