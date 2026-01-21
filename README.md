# Example of Git repository
### Get remote
```bash
#stop controlling by git
rm -rf .git 

# set up remote
git clone https://github.com/ZifengC/TrustSAR.git 
cd mlops-project
git init
git remote add origin https://github.com/ZifengC/TrustSAR.git

# verify the remote is set
git remote -v

```
### Set Up Authentication -SSH
```bash
# Generate a new SSH key (use your GitHub email)
ssh-keygen -t ed25519 -C "zchen675@uic.edu"
# 2. 启动 ssh-agent
eval "$(ssh-agent -s)"
# 3. 加载你的 key
ssh-add ~/.ssh/id_ed25519
# 4. 显示公钥并复制
cat ~/.ssh/id_ed25519.pub

ssh -T git@github.com
git remote set-url origin git@github.com:ZifengC/TrustSAR.git
```

### Change files and push
```bash
# Stage all files
git add .
git status

# Commit
git commit -m "Initial project structure for Milestone 0"

# Push to GitHub
#git branch -m master main
git push -u origin main

```
