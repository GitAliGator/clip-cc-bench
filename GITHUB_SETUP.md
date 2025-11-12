# GitHub Repository Setup Guide

Step-by-step instructions to create and push CLIP-CC-Bench to GitHub.

## Current Status

✅ Git repository initialized locally
✅ Initial commit created (90 files, 13,849 lines)
✅ .gitignore configured to exclude large files
✅ Documentation complete (README.md, SETUP.md)

## Step 1: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. **Go to GitHub**
   - Navigate to https://github.com/
   - Click the "+" icon in the top right
   - Select "New repository"

2. **Repository Settings**
   - **Repository name**: `clip-cc-bench`
   - **Description**: "Video captioning benchmark with multi-encoder evaluation framework"
   - **Visibility**: Choose Public or Private
   - **Initialize repository**:
     - ❌ DO NOT add README (we already have one)
     - ❌ DO NOT add .gitignore (we already have one)
     - ❌ DO NOT add license (add later if needed)

3. **Create Repository**
   - Click "Create repository"
   - GitHub will show you the remote URL

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI if not already installed
# On Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md

# Login to GitHub
gh auth login

# Create repository
cd /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench
gh repo create clip-cc-bench --public --source=. --remote=origin --push

# Or for private repository
gh repo create clip-cc-bench --private --source=. --remote=origin --push
```

If using GitHub CLI with `--push`, you're done! Skip to Step 4.

## Step 2: Add GitHub Remote

After creating the repository on GitHub (Option A), add it as a remote:

```bash
cd /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/clip-cc-bench.git

# Or use SSH (if you have SSH keys set up)
git remote add origin git@github.com:YOUR_USERNAME/clip-cc-bench.git

# Verify remote
git remote -v
```

## Step 3: Push to GitHub

```bash
# Rename branch to 'main' (optional, modern convention)
git branch -M main

# Push to GitHub
git push -u origin main
```

### If you encounter issues:

**Authentication Error (HTTPS)**:
- Use Personal Access Token instead of password
- Generate token at: https://github.com/settings/tokens
- Use token as password when prompted

**Authentication Error (SSH)**:
- Set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

**Large file errors**:
- Check .gitignore is working: `git status`
- Verify no large files staged: `git ls-files | xargs ls -lh | sort -k5 -h | tail`

## Step 4: Verify Upload

1. **Check GitHub**
   - Navigate to https://github.com/YOUR_USERNAME/clip-cc-bench
   - Verify all files are present
   - Check README.md displays correctly

2. **Verify Local Status**
   ```bash
   git status
   git log --oneline
   git remote show origin
   ```

## Step 5: Add Repository Metadata (Optional)

On GitHub, add:

1. **Topics/Tags** (Settings → Topics)
   - `video-captioning`
   - `benchmark`
   - `embedding`
   - `evaluation`
   - `computer-vision`
   - `nlp`
   - `pytorch`
   - `transformers`

2. **About Section**
   - Description: "Video captioning benchmark with multi-encoder evaluation framework"
   - Website: Your project website (if any)

3. **Social Preview Image**
   - Upload a preview image for social sharing

## Step 6: Add License (Optional)

If you want to add a license:

```bash
# On GitHub, go to repository → Add file → Create new file
# Name it LICENSE
# GitHub will offer license templates

# Or locally:
# Download license file (e.g., MIT, Apache 2.0)
# Add and commit
git add LICENSE
git commit -m "Add LICENSE"
git push
```

## Step 7: Add Collaboration Features

### GitHub Issues

Enable issues for bug reports and feature requests:
- Settings → Features → Issues (checkbox)

### GitHub Actions (CI/CD)

Create `.github/workflows/tests.yml` for automated testing (future):

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Run linting
        run: |
          pip install flake8
          flake8 src/
```

### Branch Protection

For collaborative development:
- Settings → Branches → Add rule
- Branch name pattern: `main`
- Require pull request reviews

## Common Commands Reference

```bash
# Check status
git status
git log --oneline --graph

# Pull latest changes
git pull origin main

# Create new branch
git checkout -b feature-name

# Add and commit changes
git add .
git commit -m "Description of changes"

# Push changes
git push origin branch-name

# View remotes
git remote -v

# View differences
git diff
git diff --staged
```

## Repository Structure on GitHub

Your repository will show:

```
clip-cc-bench/
├── .gitignore          # Excludes large files
├── README.md           # Main documentation
├── SETUP.md            # Detailed setup guide
├── LICENSE             # (Optional) License file
├── src/                # Source code
│   ├── config/        # Encoder configurations
│   ├── scripts/       # Evaluation scripts
│   └── utils/         # Utilities
├── data/              # Data structure (empty, with .gitkeep)
├── results/           # Results structure (empty, with .gitkeep)
└── slurm_scripts/     # HPC job scripts
```

## What's NOT in the Repository

These are intentionally excluded via .gitignore:

- ❌ `encoder_models/` (119GB) - Too large, download separately
- ❌ `venv/`, `venv_*/` (57GB) - Virtual environments, recreate locally
- ❌ `results/` (contents) - Generated outputs
- ❌ `data/*.json` (actual data files) - Download separately
- ❌ `.log`, `.cache`, etc. - Temporary files

Users must:
1. Clone the repository
2. Download encoder models separately
3. Create virtual environments using provided scripts
4. Provide their own data

## Sharing Your Repository

Once pushed, share with:

```
https://github.com/YOUR_USERNAME/clip-cc-bench
```

### Clone command for others:

```bash
git clone https://github.com/YOUR_USERNAME/clip-cc-bench.git
cd clip-cc-bench
# Follow SETUP.md for complete setup
```

## Next Steps After Push

1. **Update README.md** with correct GitHub URLs
2. **Add badges** (build status, license, etc.)
3. **Create releases** for stable versions
4. **Write CONTRIBUTING.md** for contributors
5. **Add GitHub Issues templates**
6. **Set up GitHub Discussions** for Q&A

## Troubleshooting

### Push rejected (non-fast-forward)

```bash
# If GitHub has commits you don't have locally
git pull --rebase origin main
git push origin main
```

### Accidentally committed large file

```bash
# Remove file from git but keep locally
git rm --cached large_file.bin
echo "large_file.bin" >> .gitignore
git add .gitignore
git commit -m "Remove large file and update .gitignore"
git push
```

### Need to rewrite history (remove sensitive data)

```bash
# Use git-filter-repo or BFG Repo-Cleaner
# See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
```

---

**You're all set!** 🎉

Your CLIP-CC-Bench repository is ready for GitHub. Follow the steps above to create and push your repository.
