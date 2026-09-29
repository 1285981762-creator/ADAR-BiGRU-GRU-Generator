# ADAR BiGRU-GRU RNA候选序列生成器

这是一个可本地运行的、带编辑效率条件的BiGRU-GRU序列生成模型。输入含目标A的RNA靶标序列
（把目标A写成`V`），模型输出5′→3′互补链候选。

模型负责生成候选，不保证输入的期望效率等于真实实验效率。推荐在生成后使用Model04或其他
已验证的编辑效率模型重新预测和排序，再进行结构、脱靶和湿实验验证。

## 仓库内容

```text
adar_bigru/                 Python包
  weights/bigru_gru_v3.pt   已训练BiGRU-GRU权重
examples/example_targets.csv
scripts/run_example.ps1     Windows示例
scripts/run_example.sh      Linux/WSL示例
tests/                      基础测试
MODEL_CARD.md               模型结构、指标和限制
```

## 1. 本地环境

要求Python 3.10或更高版本。CPU可以运行；批量生成时建议使用NVIDIA GPU。

### Windows PowerShell

```powershell
git clone <你的GitHub仓库地址>
cd <仓库目录>
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Linux或WSL

```bash
git clone <你的GitHub仓库地址>
cd <仓库目录>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

如果需要GPU版PyTorch，请先根据[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)
安装与本机系统和CUDA匹配的PyTorch，然后执行`python -m pip install -e .`。

检查安装：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
adar-bigru-generate --help
```

## 2. 输入文件

输入是UTF-8 CSV，至少包含：

```csv
target_id,target_sequence
target_1,AUGCCUVGCAUUGC
```

- `target_id`：可选；缺失时自动生成。
- `target_sequence`：必需，只能含`A/U/G/C/V`，也接受`T`并自动转为`U`。
- 每条序列必须且只能有一个`V`；`V`表示需要编辑的目标A。
- 当前权重支持的最大长度为256 nt。

## 3. 最简单的运行方法

```bash
adar-bigru-generate \
  --input examples/example_targets.csv \
  --output outputs/generated_candidates.csv
```

也可以不用控制台入口：

```bash
python -m adar_bigru.cli \
  --input examples/example_targets.csv \
  --output outputs/generated_candidates.csv
```

未指定参数时，每个靶标生成4条候选，期望效率80%，位点内百分位0.9，最低互补度90%，
guidance为2，temperature为0.8，top-k为6。

## 4. 生成指定数量和条件的序列

每个靶标生成100条候选：

```bash
adar-bigru-generate \
  --input examples/example_targets.csv \
  --output outputs/high_efficiency_candidates.csv \
  --num-sequences 100 \
  --efficiency 80 \
  --percentile 0.90 \
  --min-complementarity 0.90 \
  --guidance-scale 2 \
  --device auto
```

`--num-sequences 100`表示每个输入靶标输出100条采样记录。随机采样可能产生重复序列，
所以唯一序列数可能少于100。

## 5. 指定互补程度

例如要求最低95%互补度：

```bash
adar-bigru-generate \
  --input examples/example_targets.csv \
  --output outputs/complementarity_95.csv \
  --num-sequences 50 \
  --min-complementarity 0.95
```

还可以用`--max-edits 2`进一步把替换、插入和删除动作限制在最多2个。有效编辑上限取
“互补度允许值”和`--max-edits`中更严格的一个。低于训练时90%互补度属于分布外使用，程序会警告。

## 6. 常用参数与默认值

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--num-sequences` | 4 | 每个靶标生成的记录数 |
| `--efficiency` | 80 | 期望编辑效率条件，0–100 |
| `--percentile` | 0.90 | 期望位点内百分位，0–1 |
| `--min-complementarity` | 0.90 | 最低互补度；默认从权重读取 |
| `--max-edits` | 自动 | 可选的更严格编辑动作上限 |
| `--guidance-scale` | 2.0 | 条件引导强度，至少为1 |
| `--temperature` | 0.8 | 采样温度；0表示贪心解码 |
| `--top-k` | 6 | 每一步保留的候选动作数 |
| `--batch-size` | 64 | 推理批量大小 |
| `--seed` | 20260922 | 随机种子 |
| `--device` | auto | 自动选CUDA，否则CPU |

输入效率是生成条件，不是校准后的效率预测值。

## 7. 输出

主要输出为指定的CSV，同时生成同名`.summary.json`。CSV字段包括：

- 靶标ID、靶标序列和目标位置
- 生成的5′→3′互补链
- 期望编辑效率和位点内百分位
- 要求的最低互补度和实际互补度
- 相对严格互补链的编辑距离
- 编辑动作数和模型内部质量分数

加上`--include-strict-complement`可为每个靶标额外附加一条100%严格互补基线。

## 8. 推荐筛选流程

1. 每个靶标用高效率条件生成50–100条候选。
2. 去除重复序列，保留达到互补度要求的序列。
3. 使用Model04预测编辑效率并由高到低排序。
4. 兼顾序列多样性、结构和脱靶风险，选择少量候选进行实验。

## 9. 上传GitHub

权重约11.3 MB，低于GitHub单文件100 MB限制，可以随普通Git提交：

```bash
git init
git add .
git commit -m "Add ADAR BiGRU-GRU candidate generator"
git branch -M main
git remote add origin <你的GitHub仓库地址>
git push -u origin main
```

本包没有替项目所有者选择开源许可证。公开发布前请根据数据、代码和模型权重的授权情况，
自行添加合适的`LICENSE`文件。


靳老师在用的时候还是直接选择Linux或者wsl，windows本地化起来一堆问题，而且没有linux简洁，如果要批量跑，还是得上gpu，要求不高，4060啥的都ok，但是mac的M系列芯片上的集成gpu应该是没法用，mac就用cpu吧。