#!/usr/bin/env python
"""
Utility script to manually control document anonymization.
"""

import os
import argparse
import logging
from anonymize_mails import anonymize_documents, watch_and_anonymize
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_anonymization(watch=False, interval=60):
    """Run the anonymization process and optionally start watcher"""
    # Base directory of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define input and output directories
    documents_dir = os.path.join(base_dir, 'data', 'emails', 'documents')
    anonymized_dir = os.path.join(base_dir, 'data', 'emails', 'anonymized')
    
    # Create directories if they don't exist
    os.makedirs(documents_dir, exist_ok=True)
    os.makedirs(anonymized_dir, exist_ok=True)
    
    # Process existing documents
    logger.info(f"Processing documents from {documents_dir}")
    anonymized_files = anonymize_documents(documents_dir, anonymized_dir)
    logger.info(f"Processed {len(anonymized_files)} document(s)")
    
    # Start watching if requested
    if watch:
        logger.info(f"Starting watcher to monitor {documents_dir} for new documents")
        logger.info(f"Check interval: {interval} seconds")
        logger.info("Press Ctrl+C to stop")
        
        try:
            watch_and_anonymize(documents_dir, anonymized_dir, interval)
        except KeyboardInterrupt:
            logger.info("Watcher stopped by user")
        
def main():
    parser = argparse.ArgumentParser(description="Document Anonymization Utility")
    parser.add_argument('--watch', action='store_true', help='Watch for new documents and anonymize them')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    
    args = parser.parse_args()
    
    run_anonymization(args.watch, args.interval)

if __name__ == "__main__":
    main()
