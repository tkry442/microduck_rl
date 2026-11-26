# XC330 サーボを測って、シミュレーションを実物に合わせる(BAM)

自作の二足(36 cm・20 関節)は Dynamixel **XC330-T288** を使います。
ところが学習で使っているサーボの模型 `xc330_est.json` は、**測った値ではありません**。

- 摩擦・armature(モーターの回りにくさ)は、別のサーボ **XL330** を Rhoban が測った値をそのまま借りている
- 力の強さ(`kt`・`R`)は、カタログの「止まったときの力 0.92 N·m・何もつけないときの速さ 65 rpm(11.1 V)」から計算しただけ

この手順では、XC330 を 1 個だけ台に固定して振り子をつけ、いろいろ動かして記録し、
[BAM(Better Actuator Models)](https://github.com/Rhoban/bam) で模型の数字を当てはめます。
結果の JSON を学習の設定に差しかえれば、シミュレーションのサーボが実物に近づきます。

## いまの設定値(2026-11 時点)

| 項目 | 値 | どこ |
|---|---|---|
| サーボ | XC330-T288(20 個) | `docs/biped-design`(PC) |
| 通信 | Dynamixel Protocol 2.0、1 Mbps、TTL の 3 本線 | `microduck` の `duck-control/src/model.rs` と同じ |
| 模型 | BAM の m6(摩擦が重さ・速さで変わる、いちばん細かい形) | `xc330_est.json` |
| `kt` / `R` | 1.63 / 19.7(カタログから計算) | 同上 |
| armature・摩擦 | XL330 の測定値の借りもの | 同上 |
| ファームウェアの P(`kp_fw`) | 200(`MICRODUCK_BIPED_KP`) | `biped_task.py` |
| 電圧 | 10.5〜12.4 V でばらつかせる | 同上 |
| 通信の遅れ | 3〜6 ステップ(15〜30 ms)でばらつかせる | 同上 |

BAM が当てはめるのは **`kt`・`R`・armature・摩擦 11 個・角度のずれ `q_offset`** です。
通信の遅れとガタ(バックラッシュ)は BAM では測りません(最後の「この手順で分からないこと」)。

## 用意するもの

| もの | めやす |
|---|---|
| XC330-T288 | 1 個(できれば 2〜3 個測ると、1 個ずつの違いが分かる) |
| U2D2 | USB と Dynamixel の線をつなぐ箱。中身は FTDI という部品 |
| 電源 | 12 V(11.1〜12 V)、1 A 以上。AC アダプターか 3S のリポ。U2D2 Power Hub があるとつなぎやすい。**ラズパイから電気をとらない**(ラズパイの電源はもうぎりぎり) |
| 台 | サーボをがっちり押さえる物。万力かクランプで机に固定。腕は真下から左右 120° くらいまで回るので、そのぶん空けておく |
| 腕 | 3D プリントの棒を 3 本(軸からおもりの中心まで 100・150・200 mm)。サーボのホーンにねじで止める |
| おもり | ナットやワッシャーをまとめた物。下の表の重さを作る |
| はかり | 1 g 単位でよい(0.1 g ならなおよい) |
| ものさし | 長さを mm で測る |
| ラズパイ 3B | U2D2 を USB でつなぐ(`qq44@pi3b.local`) |

### 腕とおもりの組み合わせ(9 通り)

おもりを真横にしたときの力(重さ × 9.8 × 長さ)が、サーボの力 0.92 N·m の **半分くらいまで**になるようにしています。
重すぎると腕が上がらず、サーボも熱くなります(記録の道具は 0.75 N·m をこえると止まります)。

| 腕の長さ | 軽い | ふつう | 重い |
|---|---|---|---|
| 100 mm | 150 g | 300 g | 500 g |
| 150 mm | 100 g | 200 g | 350 g |
| 200 mm | 75 g | 150 g | 250 g |

ぴったりでなくてよいです。**作った物の本当の重さと長さ**を測って、記録のときに入れます。

- `--mass`: おもりだけの重さ(kg)。ねじも含める
- `--arm-mass`: 腕の棒だけの重さ(kg)
- `--length`: サーボの軸の中心から、おもりの中心までの長さ(m)

## 1. ラズパイの準備(1 回だけ)

PC の PowerShell からラズパイに入ります。

```bash
ssh qq44@pi3b.local
```

ここからはラズパイの中です。道具を入れます(numpy と rustypot だけ。数分)。

```bash
python3 -m venv ~/sysid && ~/sysid/bin/pip install numpy rustypot
```

記録の道具を GitHub から持ってきます。

```bash
curl -fsSLo ~/record_xc330.py https://raw.githubusercontent.com/tkry442/microduck_rl/claude/project-thread-3c4vdm/scripts/sysid/record_xc330.py
```

U2D2 をラズパイの USB にさして、名前を確かめます(ふつうは `/dev/ttyUSB0`)。

```bash
ls /dev/ttyUSB*
```

U2D2 は答えを 16 ms ためてから送る設定になっているので、1 ms にします(さし直すたびに必要)。

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

## 2. サーボの準備(サーボ 1 個につき 1 回)

サーボ 1 個だけを U2D2 につなぎ、12 V の電源を入れます。買ったままのサーボは ID 1・57600 bps です。

```bash
~/sysid/bin/python ~/record_xc330.py setup
```

速さを 1 Mbps、返事の待ち時間を 0、位置で動かすモードにします(サーボの中に書きこまれ、電源を切っても残ります。ロボットでも同じ設定を使います)。
`setup OK` と出れば成功です。型番の数字とファームウェアの版も出るので、メモしておいてください。

## 3. 「真下」を 0° として覚えさせる

腕(おもりなしでよい)をつけて、まっすぐ下にたらします。

```bash
~/sysid/bin/python ~/record_xc330.py zero --logdir ~/sysid_raw
```

力を抜いて 3 秒測り、`~/sysid_raw/zero_id1.txt` に保存します。
「範囲に入らない」と出たら、腕をホーンから外して、ホーンの印が真下を向くようにつけ直してから、もう一度やります。

手で腕を動かして、角度が合っているか見ます(Ctrl-C で終わり)。

```bash
~/sysid/bin/python ~/record_xc330.py check --logdir ~/sysid_raw
```

真下で 0°、真横で +90° か −90° になれば OK です(どちら向きが + でもかまいません)。
右の `reads/s` が 300 より小さいときは、latency_timer の設定を見直します。

## 4. 記録する(1 組み合わせ 約 5 分 × 9)

おもりをつけて、重さと長さを入れて記録します。例: 150 mm の腕(12 g)に 200 g のおもり。

```bash
~/sysid/bin/python ~/record_xc330.py record --logdir ~/sysid_raw --mass 0.200 --arm-mass 0.012 --length 0.150
```

1 回で、P の強さ 5 通り(50・100・200・300・400)× 動き 4 通り = **20 本**を記録します。

| 動き | 何をするか | 何が分かるか |
|---|---|---|
| `sin_sin` | 大きくゆっくり + 細かく速く | いろいろな速さ |
| `lift_and_drop` | −90° まで上げて、**力を抜いて落とす** | 力を抜いたときの摩擦 |
| `up_and_down` | ゆっくり +90° へ行って少し戻る | 止まりかけの摩擦、重さで変わる摩擦 |
| `sin_time_square` | だんだん速くなる往復 | 速いときのモーターの限界 |

- 腕は勢いよく回り、落ちます。**手や顔を近づけない**でください。
- サーボが 55 °C より熱いと、冷えるまで待ちます。70 °C をこえたら止まります。
- 1 本ごとに `step 3.1 ms` のように記録の間かくが出ます。**5 ms より小さい**ことを確かめます。
- 終わると力を抜きます。Ctrl-C で止めても力を抜きます。

おもりと腕を取りかえて、9 通りぶんくり返します(合わせて 180 本、1 時間くらい)。
できれば `--length` は 1 mm 単位、`--mass` は 1 g 単位で正しく入れてください。ここがずれると、BAM はそのずれを摩擦のせいにします。

## 5. 記録を PC へ移す

PC の PowerShell で(ラズパイからは出て):

```powershell
scp -r qq44@pi3b.local:~/sysid_raw C:\Users\tkry3\dev\microduck\sysid_data\
```

## 6. 当てはめる(PC の WSL、数時間)

ここからは WSL の `microduck_rl` で行います(このブランチの `scripts/sysid/` と `src/mjlab_microduck/actuator/xc330.py` が要ります)。

まず、記録の時刻を 5 ms ごとにそろえます。

```bash
uv run python scripts/sysid/bam_xc330.py process --raw /mnt/c/Users/tkry3/dev/microduck/sysid_data/sysid_raw --logdir sysid_data/processed --dt 0.005
```

次に当てはめます。P = 300 の記録は当てはめに使わず、**答え合わせ用**に取っておきます。

```bash
uv run --with optuna --with cmaes --with wandb python scripts/sysid/bam_xc330.py fit --actuator xc330 --model m6 --logdir sysid_data/processed --validation_kp 300 --trials 5000 --workers 8 --output sysid_data/xc330_m6.json
```

- `--trials` は 1 つの作業あたりの回数です。`--workers 8` なら 8 × 5000 = 4 万回ためします。PC のコアの数より多くしないでください。
- 途中でも数秒ごとに `xc330_m6.json` が書きかわるので、いつ止めても使えます。
- `Best score` は「シミュレーションと実物の角度のずれの平均」(rad)です。0.02 なら約 1.1°。
- 答え合わせのずれ(`val_best_value`)が当てはめのずれより大きく離れていくなら、数字を覚えこみすぎています。

にせのサーボで試したときは、180 本ではなく 36 本・1500 回で約 3 分、ずれ 0.022 rad でした。本番は 180 本なので、1 回の計算が約 5 倍かかります。

## 7. 目で確かめる

実物(線)とシミュレーション(点線)を重ねて表示します。

```bash
uv run python scripts/sysid/bam_xc330.py plot --actuator xc330 --logdir sysid_data/processed --params sysid_data/xc330_m6.json --sim
```

`lift_and_drop` の落ち方と、`sin_time_square` の速いところが重なっていれば合っています。
今の借りものの模型 `xc330_est.json` も `--params` に並べて入れると、どれだけ良くなったか比べられます。

## 8. 学習に使う

1. `sysid_data/xc330_m6.json` を、`xc330_est.json` と同じフォルダ(`src/mjlab_microduck/robot/microduck/`)に写す
2. 学習のときに `MICRODUCK_BIPED_MOTOR=xc330_m6` をつける(`biped_task.py` はこの名前の JSON を読みます)
3. `MICRODUCK_BIPED_KP` を、**ロボットで実際に使う P の値**にする

JSON の中は `"actuator": "xc330"` です。`mjlab_microduck.actuator` を読みこむと `xc330` が登録されるので、そのまま読めます。
もし手元の WSL のコードにこのブランチの変更がまだ入っていなければ、JSON の `"actuator"` を `"xl330"` に書きかえても同じ動きになります
(ちがうのは電流の上限 1.75 A だけで、XC330 は抵抗が大きいので 1 A にも届かず、効きません)。

## この手順で分からないこと

- **通信の遅れ**: BAM は「指令を出したらすぐ届く」として当てはめます。学習の 15〜30 ms の遅れのばらつきはそのまま残します。ロボット全体で 20 個を読み書きしたときの本当の遅れは、別に測ります。
- **ガタ(バックラッシュ)**: 振り子はいつも重さで片側に押しつけられているので、ガタはほとんど出ません。
- **1 個ずつの違い**: 1 個だけ測ると、その 1 個の値です。2〜3 個測ってくらべると、学習でどれくらいばらつかせればよいかが分かります。
- **熱**: 熱くなるとモーターの抵抗が上がります。記録は 55 °C 以下から始めます。

## 道具の中身(調べたいとき)

| ファイル | 役目 |
|---|---|
| `scripts/sysid/record_xc330.py` | ラズパイで記録する。numpy と rustypot だけで動く。`--fake 模型.json` で、サーボなしで PC で練習できる |
| `scripts/sysid/bam_xc330.py` | BAM の `process`・`fit`・`plot`・`mae` に `xc330` を登録してから動かす |
| `src/mjlab_microduck/actuator/xc330.py` | BAM 用の XC330 の定義。XL330 と同じファームウェアの式で、`kt`・`R` の探す範囲だけ広げた |
| `tests/test_xc330_sysid.py` | 記録の形・角度の 0・向き・単位が BAM と合っているかの試験 |

BAM の XL330 用の記録の道具(`bam.dynamixel.record`)は、腕の重さを `arm-mass` という名前で書くので、BAM の読む側(`arm_mass`)に届かず、腕の重さが 0 として扱われます。
`record_xc330.py` は `arm_mass` で書きます。
