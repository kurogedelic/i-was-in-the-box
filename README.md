# 『箱の中に居たのは私だった。』

**Leo Kuroshita**  
**2025 / CC BY-NC-SA 4.0**

DAWレスで制作された実験音楽アルバム。全10トラック完成。

## 収録トラック

1. **偏頭痛／刹那的な感覚の再現** — 5音源の動的グラニュラー合成。カオス関数駆動の粒子位置、20セグメント進行、FFTスペクトル凍結
2. **ラッピングされた君の会話** — Csoundユークリッドリズム生成。Zap合成(12kHz→20Hz)、280→400BPMテンポ加速、164秒急停止
3. **不確定の庭** — 16×16セルオートマトン音楽生成。Conway変形則、ペンタトニック・マッピング、Cマイナー/ドリアン和声進行
4. **Drosera's Song** — Rosenberg声門パルス+カスケード・フォルマント合成。科学的に正確な食虫植物解説童謡
5. **応力–ひずみ曲線** — 材料試験CSVデータ音響化。弾性→降伏→塑性→破断の共振周波数変化、300MPa以上で歪み
6. **Choral Induction Protocol** — 8声ポリフォニック・フォルマント合成。スレンドロ音程、ガムラン・コテカン構造、Haas効果
7. **Constellation - Phase 1 Group 2** — 合成衛星データ+NASA公開データの音響化。高度→周波数、軌道傾斜角→変調マッピング
8. **アシッド・テクノの印象** — Web Audio API TB-303モデリング。共振フィルタースイープ、リアルタイムパターン生成 (acid-test500.vercel.app)
9. **夜の輝く湖水 (Avonlea)** — norns環境応答型アンビエント。月相・光条件反応、赤毛のアン着想 (github.com/kurogedelic/avonlea)
10. **以下、SCSIディスクが回答します。** — 10000RPM物理モデル。166.7Hz基本周波数、ベアリング共振2-8kHz、シーク動作500-3000Hzチャープ

## プロジェクト構成

```
InTheBox/
├── tracks/           # 各トラック制作用ソースコード
│   ├── EMRSP/        # トラック1: 偏頭痛／刹那的な感覚の再現
│   ├── Your_Wrapped_Conversation/ # トラック2: ラッピングされた君の会話
│   ├── field/        # トラック3: 不確定の庭
│   ├── drosera/      # トラック4: Drosera's Song
│   ├── stress-strain_curve/ # トラック5: 応力–ひずみ曲線
│   ├── voices/       # トラック6: Choral Induction Protocol
│   ├── constellation/ # トラック7: Constellation - Phase 1 Group 2
│   └── scsi_disk/    # トラック10: SCSIディスク
├── release/          # リリース版（WAV 48kHz/24bit）
├── masters/          # マスター版（WAV -14 LUFS）
├── mastering/        # マスタリング・スクリプト
├── bounces/          # 制作中間ファイル
├── docs/            # ドキュメント・ライセンス
└── acidMore/        # トラック8制作用Webアプリ
```

## ファイル形式

### リリース版 (`release/`)
- **WAV**: 24bit @ 48kHz（配布／配信用の基準フォーマット）
- **メタデータ**: 曲名・アーティスト等を埋め込み（ffmpeg）
- **音圧目標**:
  - Apple Music: −16 LUFS / True Peak ≤ −1 dBTP（Sound Check想定）
  - Bandcamp: −14 LUFS / True Peak ≤ −1 dBTP（正規化なし想定）

### マスター版 (`masters/wav/`)
- **WAV**: 24bit @ 48kHz 
- **用途**: アーカイブ・再マスタリングのベース

## 制作・再現

### 依存関係
- **Python 3.8+** + NumPy, SciPy
- **SuperCollider** (応力–ひずみ曲線)
- **FAUST** (SCSIディスク)
- **norns** (Avonlea)
- **FFmpeg** (マスタリング)

### 実行方法
```bash
# 全トラック再ビルド（個別実行）
cd tracks/EMRSP && python emrsp_granular.py
cd tracks/drosera && python drosera_with_piano.py
# ... 他トラック

# マスタリング（48kHz/24bit WAVを dist/ に出力）
cd mastering && python master_final_album.py

# LUFS正規化（配信先に合わせて出力先を分ける例）
# Apple Music向け（−16 LUFS / −1 dBTP）
python mastering/normalize_lufs.py \
  --input-dir dist --output-dir dist_apple_music --target-lufs -16

# Bandcamp向け（−14 LUFS / −1 dBTP）
python mastering/normalize_lufs.py \
  --input-dir dist --output-dir dist_bandcamp --target-lufs -14
```

### 再現性
- 乱数には `--seed` を付与
- 各トラック個別README参照
- 出力は 48kHz/24bit WAV。配信向けLUFSは正規化スクリプトで明示指定（例: Apple Music −16 / Bandcamp −14）。

## ライセンス

- **コード**: GPL-3.0 (`docs/LICENSE`)
- **音源・アート**: CC BY-NC-SA 4.0 (`docs/LICENSE-AUDIO.md`)
- **法務詳細**: `docs/LEGAL.md`

## クレジット

- **作者**: Leo Kuroshita
- **制作年**: 2025
- **形式**: Experimental Music Album - DAWless Production
- **依存データ**: Starlink/TLE等の出典は `docs/LEGAL.md` 参照

詳細は `docs/CREDITS.md` を参照してください。
# i-was-in-the-box
