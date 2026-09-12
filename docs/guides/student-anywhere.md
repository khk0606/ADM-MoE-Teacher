# 30개 방 Student anywhere 확장

## 이번 변경

모든 30개 방에 Student 목적 prompt `sit_anywhere_v1` = **Sit on something**을 추가합니다.
단순히 뷰어 표시를 추가한 것이 아닙니다. 실제 text feature, 학습 레코드, 목적 제약 없는 목표값, 평가 조건을 추가합니다.

- 기존 목적 조건 270개를 그대로 유지하고 anywhere 180개를 추가합니다.
- train g0/g1: 기존180 + 새120 = **300개**.
- eval g2: 기존90 + 새60 = **150개**. 모든 방은 개발 데이터이며 unseen-room 평가가 아닙니다.
- 방/세대가 같으면 기존 sit action의 **동일한 saved1578 Teacher A**를 재사용합니다. Teacher 재학습·재생성·AMDM 실행 없음.
- 모델 입력/구조, 후보 점수 0.7R+0.3H, softmax 온도0.1, 최종 A_raw×w는 유지합니다. 곱셈형 R×H 제안은 적용하지 않습니다.
- anywhere에서는 특정 목적 가구를 지정하지 않으므로 두 유효 앉기 후보 모두 R 목표1, purpose class/mask 목표0입니다.
  History 점수·공간 지지 영역 목표는 기존과 같습니다. 모든 가구/바닥을 앉기 후보로 간주하는 것은 아닙니다.
- 기존 목적 조건의 Relation/History/지원 영역 목표는 바꾸지 않습니다. 기존 목적에서 항상 history가 선택을 뒤집도록 만드는 변경도 아닙니다.

## 실제 text feature

원래와 동일한 OpenAI CLIP ViT-B/32 가중치를 SHA256으로 확인합니다.
기존4문장은 저장된 feature를 그대로 사용하고, 새 `Sit on something`은 실제로 인코딩합니다.
새 encoder로 기존4문장도 비교하여 같은 feature 공간인지 검사합니다.
공식 JIT archive를 읽고 eager model로 재구성하며 임의 torch.load fallback은 없습니다. 자동 다운로드나 패키지 업그레이드는 하지 않습니다.

기본 경로는 `~/.cache/clip/ViT-B-32.pt`입니다. 다른 곳이면 preflight와 train 모두 `--clip-weights /실제/ViT-B-32.pt`를 지정하세요.
CLIP 구현 참고: https://github.com/openai/CLIP/blob/main/clip/clip.py

## 실행

ZIP을 사용자 AMDM 루트에 복사하고 기존 afford 환경에서:

```bash
cd ~/AMDM &&
unzip -n small_room30_student_anywhere_v1.zip &&
bash scripts/small_room30/student_anywhere.sh preflight &&
bash scripts/small_room30/student_anywhere.sh train
```

초기 모델은 완료된 `outputs/small_room30_student_competition_cpu_eval01/summary.json`입니다.
그 모델을 복사해 **기존 목적 + anywhere를 함께 2,000 step 추가 학습**합니다. 처음부터 3,600 step을 다시 돌리지 않습니다.
기본 학습률3e-4. 통과했던 조건도 가중치 변경으로 달라질 수 있으므로 기존90조건을 직전 모델과 반드시 비교합니다.
Teacher와 원본 competition 체크포인트는 변경하지 않습니다. 새 출력 디렉터리만 만듭니다.
평가는 처음부터 **CPU 단일 스레드**로 실행하고 기존 엄격한 후보 위치/분기 일관성 검사를 유지합니다.

초기 경로가 다르면 `--initial-summary outputs/실제_복구_출력/summary.json`을 양쪽 명령에 붙이세요.
중복 출력 디렉터리는 거부합니다. 재시도 시 `OUTPUT=outputs/새로운_이름`을 지정하세요.

## 확인

```bash
PORT=8100 bash scripts/small_room30/student_anywhere.sh view
```

8100 포트를 전달하고 제목 `Anywhere + Purpose Student`를 확인하세요.

1. 방 선택 → 목적 prompt `sit_anywhere_v1` 고정 → history 출처만 변경.
2. R은 두 후보 모두 높게 유지되는지, H 및 최종 우세 후보가 history에 따라 달라지는지 확인.
3. 선택 후보 주변 바닥 지지 영역도 함께 확인.
4. 기존 목적 prompt도 다시 확인. `이전 결과 / Teacher / 새 결과`로 이전 competition 결과와 비교.

anywhere는 이전 Student 결과가 없으므로 비교 화면 왼쪽을 **Teacher A — 이전 anywhere Student 없음**으로 명시합니다.
이를 기존 Student가 생성한 anywhere 결과로 오해하면 안 됩니다.
history 근처부터 경로를 생성하는 기능은 아니며, 두 입력 history가 같은 후보를 선호하면 무조건 뒤집도록 강제하지 않습니다.
실제 관측 기반 목표 감사에서 25개 방은 history에 따라 우세 후보가 바뀌며, 0201–0205는 두 history 모두 같은 후보를 선호합니다.
해당 다섯 방은 입력 쌍만으로 선택 전환을 요구하는 검증에 적합하지 않습니다. 뷰어에 목표 전환 예상 여부를 표시합니다.
`comparison.json`의 실패 수/전환 실패를 보고 시각 확인하세요. 자동 승인 없음.

## 검증 한계

로컬에서 실제 geometry/history/labels와 구조·학습·평가·뷰어 연결을 검사합니다.
사용자의 실제 full1578 캐시, 최신 학습 가중치, CLIP 가중치는 로컬에 없으므로 실제 feature 생성은 GPU 호스트의 preflight에서 검사합니다.
합성 fixture의 성공을 실제150조건의 학습 성능으로 주장하지 않습니다. 실제 재학습 및 기존 통과 조건 보존은 사용자 결과에서 확인해야 합니다.
