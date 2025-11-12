#!/bin/bash
# Convenient aliases for testing
# Add these to your ~/.bashrc

alias cdimg='cd /home/du2/22CS30064/img2img-turbo'
alias test-quick='cd /home/du2/22CS30064/img2img-turbo && python simple_test.py'
alias test-full='cd /home/du2/22CS30064/img2img-turbo && python test_pix2pix_turbo.py'
alias check-jobs='squeue -u $USER'
alias watch-jobs='watch -n 5 squeue -u $USER'

echo "Aliases loaded! Use:"
echo "  cdimg       - Go to img2img-turbo directory"
echo "  test-quick  - Quick test command"
echo "  test-full   - Full test command"
echo "  check-jobs  - Check your SLURM jobs"
echo "  watch-jobs  - Watch jobs with auto-refresh"
