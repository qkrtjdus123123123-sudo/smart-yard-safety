# Streamlit Cloud 배포 가이드

Branch / Main file path가 없다고 나오는 경우, **GitHub에 코드가 실제로 올라가 있지 않을 때** 발생합니다. 아래 순서대로 진행하세요.

---

## 1단계: GitHub 저장소 만들기

1. [github.com](https://github.com) 로그인
2. 우측 상단 **+** → **New repository**
3. **Repository name**: `smart-yard-safety` (원하는 이름)
4. **Public** 선택
5. **Create repository** 클릭 (README 추가 안 해도 됨)

---

## 2단계: 프로젝트를 Git으로 올리기 (가장 중요)

**이 단계를 하지 않으면** Streamlit Cloud에서 Branch/파일을 찾을 수 없습니다.

### 2-1. 터미널에서 프로젝트 폴더로 이동

```powershell
cd c:\smart-yard-project
```

### 2-2. Git 초기화 (이미 되어 있으면 생략)

```powershell
git init
```

### 2-3. 올릴 파일만 추가

**필수 파일만** 넣어야 합니다. `data` 폴더, PDF/PPT, 추출 스크립트는 제외해도 됩니다.

```powershell
git add app.py
git add requirements.txt
git add .streamlit
```

`.streamlit` 폴더가 없으면:

```powershell
git add app.py
git add requirements.txt
```

### 2-4. 커밋

```powershell
git add .
git commit -m "Streamlit Smart Yard app"
```

일부 환경에서는 `git add .` 대신 아래만 해도 됩니다.

```powershell
git add app.py requirements.txt
git commit -m "Streamlit Smart Yard app"
```

### 2-5. GitHub 저장소와 연결

GitHub에서 만든 저장소 주소로 바꿉니다. `본인아이디` / `저장소이름` 부분만 수정하세요.

```powershell
git remote add origin https://github.com/본인아이디/smart-yard-safety.git
```

예: `https://github.com/honggil-dong/smart-yard-safety.git`

### 2-6. 브랜치 이름 확인 후 푸시

최근 GitHub는 기본 브랜치가 **main**입니다.

```powershell
git branch -M main
git push -u origin main
```

로그인 창이 뜨면 GitHub 계정으로 인증합니다.  
**이 푸시가 끝나야** GitHub 웹에서 `main` 브랜치와 `app.py` 파일이 보입니다.

---

## 3단계: Streamlit Cloud에서 배포

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. **Sign in with GitHub** → 권한 허용
3. **New app** 클릭
4. 아래처럼 **정확히** 입력:

   | 항목 | 입력값 |
   |------|--------|
   | **Repository** | `본인아이디/smart-yard-safety` (드롭다운에서 선택) |
   | **Branch** | `main` (드롭다운에서 선택) |
   | **Main file path** | `app.py` (프로젝트 **최상단**에 있을 때) |

5. **Deploy!** 클릭
6. 2~5분 정도 지나면 **Your app is live!** 와 함께 공개 URL이 생성됩니다.

---

## Branch / Main file path가 없다고 나올 때

- **Branch가 없다고 나오면**
  - GitHub 저장소 페이지에서 **상단 브랜치 이름**이 `main`인지 `master`인지 확인
  - Streamlit Cloud의 Branch에는 그 이름을 **그대로** 입력 (보통 `main`)
  - `git push`를 **한 번도 안 했다면** 브랜치가 없으므로, **2단계 2-6** 다시 실행

- **Main file path가 없다고 나오면**
  - GitHub 저장소 페이지에서 **파일 목록**에 `app.py`가 **저장소 루트(맨 위)** 에 있는지 확인
  - `app.py`가 **폴더 안**에 있으면 경로를 그에 맞게 입력 (예: `src/app.py`)
  - 루트에 `app.py`가 있으면 **`app.py`** 만 입력 (앞에 `/` 없이)

---

## requirements.txt

배포에 필요한 라이브러리는 프로젝트의 **requirements.txt** 에 이미 넣어 두었습니다.  
추가로 수정한 뒤에는 `git add requirements.txt` → `git commit` → `git push` 한 번 더 하면 됩니다.

---

## 요약 체크리스트

- [ ] GitHub에 새 저장소 생성
- [ ] `git init` → `git add app.py requirements.txt` → `git commit`
- [ ] `git remote add origin https://github.com/본인아이디/저장소이름.git`
- [ ] `git branch -M main` → `git push -u origin main`
- [ ] GitHub 웹에서 `main` 브랜치와 `app.py` 보이는지 확인
- [ ] share.streamlit.io → New app → Repository/Branch/Main file path 입력 → Deploy
