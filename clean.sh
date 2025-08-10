#!/bin/bash
# Complete Project Cleanup Script
# ===============================
#
# Comprehensive cleanup of cache, temporary files, empty files/directories, 
# and project-specific artifacts. Keeps the project optimized and organized.
#
# Features:
# - Removes Python cache and compiled files
# - Cleans IDE/editor temporary files  
# - Removes empty files and directories
# - Eliminates placeholder and test files
# - Project-specific cleanup for robotics files
# - Preserves git repository and important files

PROJECT_ROOT="$(dirname "$0")"
cd "$PROJECT_ROOT"

echo "🧹 Complete Project Cleanup Starting..."
echo "=========================================="

# 1. Remove Python cache
echo "🐍 Cleaning Python cache..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null

# 2. Remove IDE/Editor cache  
echo "💻 Cleaning IDE cache..."
rm -rf .vscode/settings.json 2>/dev/null
find . -name ".DS_Store" -delete 2>/dev/null
find . -name "Thumbs.db" -delete 2>/dev/null
find . -name "desktop.ini" -delete 2>/dev/null

# 3. Remove temporary files
echo "🗂️  Cleaning temporary files..."
find . -name "*.tmp" -delete 2>/dev/null  
find . -name "*.temp" -delete 2>/dev/null
find . -name "*~" -delete 2>/dev/null
find . -name "*.bak" -delete 2>/dev/null
find . -name "*.swp" -delete 2>/dev/null
find . -name "*.swo" -delete 2>/dev/null

# 4. Remove log files
echo "📝 Cleaning log files..."
find . -name "*.log" -delete 2>/dev/null
find . -name "*.out" -delete 2>/dev/null

# 5. Remove empty files and directories
echo "📁 Cleaning empty files and directories..."
find . -type f -empty -not -path "./.git*" -delete 2>/dev/null
find . -type d -empty -not -path "./.git*" -delete 2>/dev/null

# 6. Remove common empty/placeholder files
echo "🗃️  Removing placeholder files..."
find . -name ".gitkeep" -delete 2>/dev/null
find . -name ".keep" -delete 2>/dev/null
find . -name "placeholder.txt" -delete 2>/dev/null
find . -name "README.txt" -size 0 -delete 2>/dev/null
find . -name "*.md" -size 0 -delete 2>/dev/null

# 7. Remove duplicate empty directories (run twice to catch nested empties)
echo "🔄 Second pass - removing nested empty directories..."
find . -type d -empty -not -path "./.git*" -delete 2>/dev/null

# 6. Clean OpenCV cache if exists
echo "👁️  Cleaning OpenCV cache..."
rm -rf ~/.opencv_cache 2>/dev/null

# 7. Clean Webots cache if exists
echo "🤖 Cleaning Webots cache..."
rm -rf ~/.webots 2>/dev/null

# 8. Project-specific cleanup
echo "🎯 Project-specific cleanup..."
# Remove test output files
find . -name "test_*.png" -delete 2>/dev/null
find . -name "debug_*.jpg" -delete 2>/dev/null
find . -name "output_*.txt" -delete 2>/dev/null
# Remove compiled Python files in specific locations
rm -f controllers/*/*.pyc 2>/dev/null
rm -rf controllers/*/.__pycache__ 2>/dev/null
# Remove any leftover marker files if they exist
find . -name "marker_*.png" -delete 2>/dev/null 2>/dev/null
find . -name "aruco_test_*.png" -delete 2>/dev/null

# 9. Final cleanup pass for any remaining empty items
echo "🏁 Final cleanup pass..."
find . -type f -empty -not -path "./.git*" -not -name ".gitignore" -delete 2>/dev/null
find . -type d -empty -not -path "./.git*" -delete 2>/dev/null

echo ""
echo "✅ Complete cleanup finished!"
echo ""
echo "🧹 Cleaned items:"
echo "  • Python cache files (*.pyc, __pycache__)"
echo "  • IDE/Editor cache (.DS_Store, Thumbs.db, etc.)"
echo "  • Temporary files (*.tmp, *.temp, *~, *.bak, *.swp)"
echo "  • Log files (*.log, *.out)"
echo "  • Empty files and directories"
echo "  • Placeholder files (.gitkeep, .keep, etc.)"
echo "  • Test output files (test_*.png, debug_*.jpg)"
echo "  • Project-specific temporary files"
echo ""
echo "📊 Current project size:"
du -sh . 2>/dev/null || echo "Could not calculate project size"

echo ""
echo "📂 Directory structure after cleanup:"
find . -type d -not -path "./.git*" | head -20 | sort

echo ""
echo "💡 Tip: Run this script regularly with: ./clean.sh"
echo "⚡ Project is now optimized and clean!"
