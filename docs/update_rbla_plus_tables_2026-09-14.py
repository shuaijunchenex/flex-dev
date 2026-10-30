"""Update the supplied LaTeX RBLA+ cells from the latest local RankScale logs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/94804/.codex/attachments/768663cd-01ac-4573-bdd0-0fe261590940/pasted-text.txt')
RESULTS = ROOT / 'src/test/experiment_results'
OUT = ROOT / 'design'
LABELS = {'cifar10': 'CIFAR-10', 'cinic10': 'CINIC-10', 'svhn': 'SVHN',
          'agnews': 'AG News', 'imdb': 'IMDB', 'mnli': 'MNLI', 'mrpc': 'MRPC',
          'qqp': 'QQP', 'rte': 'RTE', 'sst2': 'SST-2'}
NAME = re.compile(r'^(?P<dataset>\w+)_dirichlet_0_4_rank_(?P<assignment>correct|mismatch)_rbla_plus_rank_scale_-train-(?P<timestamp>\d{8}_\d{6})-[^.]+\.csv$')
ROW = re.compile(r'^(?P<label>' + '|'.join(re.escape(x) for x in LABELS.values()) + r')\n(?P<body>(?:&[^\n]*\n){6}&[^\n]*\\\\)', re.M)
TABLE = re.compile(r'\\begin\{table\*\}.*?\\end\{table\*\}', re.S)
NUMBER = re.compile(r'\d+\.\d+')


def rank_values(values):
    ordered = sorted(values)
    return [(ordered.index(v) + 1 + len(ordered) - ordered[::-1].index(v)) / 2 for v in values]


def correlation(xs, ys):
    xs, ys = rank_values(xs), rank_values(ys)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))


def read_run(path):
    match = NAME.fullmatch(path.name)
    assert match
    content = path.read_bytes()
    lines = content.decode('utf-8-sig').splitlines()
    config = json.loads(lines[0].split(',', 1)[1])
    effective = config['effective_yaml']
    server = effective['server_yaml']
    assert config['aggregation'] == 'sp_plus'
    assert server['strategy']['strategy_name'] == 'rbla_plus_rank_scale'
    clients = sorted(effective['client_yamls'], key=lambda x: x['index'])
    assert len(clients) == 10
    assert all(c['yaml']['strategy']['strategy_name'] == 'rbla_plus_rank_scale' for c in clients)
    ratios = config['rank_ratio_list']
    assert config['effective_client_rank_ratios'] == ratios
    assert [c['yaml']['nn_model']['rank_ratio'] for c in clients] == ratios
    assert config['dataset'] == match['dataset']
    assert server['data_distribution']['use'] == f"{match['dataset']}_dirichlet_0_4"
    volumes = [sum(row) for row in next(iter(server['data_distribution']['custom_define'].values()))]
    rho = correlation(volumes, ratios)
    assert rho > .9 if match['assignment'] == 'correct' else rho < -.9
    header = next(i for i, line in enumerate(lines) if line.startswith('round,accuracy,'))
    rows = list(csv.DictReader(lines[header:]))
    rounds = [int(r['round']) for r in rows]
    assert rounds == list(range(len(rows))), path
    assert all(math.isfinite(float(r['accuracy'])) and 0 <= float(r['accuracy']) <= 1 for r in rows)
    expected_last = int(config['general']['training_rounds'])
    assert expected_last == 100
    assert rounds[-1] <= expected_last
    peak = max(rows, key=lambda r: Decimal(r['accuracy']))
    percentage = Decimal(peak['accuracy']) * 100
    formatted = format(percentage.quantize(Decimal('.01'), rounding=ROUND_HALF_UP), '.2f')
    return {**match.groupdict(), 'label': LABELS[match['dataset']], 'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(content).hexdigest(), 'records': len(rows),
            'last_round': rounds[-1], 'expected_last_round': expected_last,
            'complete': rounds == list(range(expected_last + 1)), 'peak_round': int(peak['round']),
            'peak_accuracy_percent': str(percentage), 'display_accuracy': formatted,
            'rank_volume_spearman': rho}


def numeric_cells(body):
    return [NUMBER.search(line).group() for line in body.splitlines()]


def main():
    # Select the most recent run for each identity, never the maximum across runs.
    candidates = defaultdict(list)
    for path in RESULTS.glob('*.csv'):
        match = NAME.fullmatch(path.name)
        if match:
            candidates[(match['dataset'], match['assignment'])].append(path)
    selected = {}
    for key, paths in candidates.items():
        path = max(paths, key=lambda p: NAME.fullmatch(p.name)['timestamp'])
        run = read_run(path)
        run['candidate_count'] = len(paths)
        selected[(run['label'], run['assignment'])] = run
    assert len(selected) == 12

    original = SOURCE.read_text(encoding='utf-8-sig')
    original_tables = TABLE.findall(original)
    assert len(original_tables) == 2
    changes = []

    def update_table(match):
        table = match.group()
        assignment = 'mismatch' if r'\label{tab:mismatched-rank-assignment}' in table else 'correct'

        def update_row(row):
            run = selected.get((row['label'], assignment))
            if run is None:
                return row.group()
            old_values = numeric_cells(row['body'])
            values = old_values[:]
            values[2] = run['display_accuracy']
            maximum = max(Decimal(x) for x in values[1:])
            rendered = [values[0]]
            for index, value in enumerate(values[1:], 1):
                cell = r'\best{' + value + '}' if Decimal(value) == maximum else value
                if index == 2 and not run['complete']:
                    cell += r'\ensuremath{^{\dagger}}'
                rendered.append(cell)
            changes.append({**run, 'original_accuracy': old_values[2],
                            'change_percentage_points': str(Decimal(values[2])-Decimal(old_values[2]))})
            return row['label'] + '\n' + '\n'.join('& ' + x for x in rendered) + r' \\'

        updated = ROW.sub(update_row, table)
        partial = [v for (_, a), v in selected.items() if a == assignment and not v['complete']]
        note = ('RBLA+ entries for CIFAR-10, CINIC-10, SVHN, AG News, IMDB, and MNLI use the latest RankScale logs; '
                'the other RBLA+ entries are retained from the supplied table.')
        if partial:
            detail = '; '.join(f"{r['label']}: logged rounds 0--{r['last_round']} of planned 0--{r['expected_last_round']}" for r in partial)
            note += r' $\dagger$ Highest accuracy observed in an incomplete log (' + detail + ').'
        updated = updated.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + r'\par\vspace{2pt}' + '\n' + r'\parbox{\textwidth}{\scriptsize\raggedright ' + note + '}')
        return updated

    result = TABLE.sub(update_table, original)
    assert len(changes) == 12
    for old_table, new_table in zip(original_tables, TABLE.findall(result)):
        old_rows = list(ROW.finditer(old_table))
        new_rows = list(ROW.finditer(new_table))
        assert len(old_rows) == len(new_rows) == 10
        for old, new in zip(old_rows, new_rows):
            assert old['label'] == new['label']
            old_numbers, new_numbers = numeric_cells(old['body']), numeric_cells(new['body'])
            assert old_numbers[:2] == new_numbers[:2]
            assert old_numbers[3:] == new_numbers[3:]
        # All multi-seed rows and their original bold formatting stay unchanged.
        assert old_table.split('\nCIFAR-10')[0] == new_table.split('\nCIFAR-10')[0]
    assert result.count(r'\ensuremath{^{\dagger}}') == 2
    assert result.count('{') == result.count('}')

    tex = OUT / 'rbla_plus_tables_updated_2026-09-14.tex'
    outputs = {str(tex): result}
    manifest = {'as_of': '2026-09-14', 'selection': 'latest logged timestamp per dataset and assignment; max accuracy across observed rounds',
                'partial_policy': 'include observed peak with dagger, explicitly authorized by user',
                'source_table': str(SOURCE), 'changes': changes,
                'unchanged': ['MNIST/FMNIST/KMNIST/QMNIST all alphas and both assignments', 'MRPC/QQP/RTE/SST-2 both assignments']}
    outputs[str(OUT / 'rbla_plus_tables_update_sources_2026-09-14.json')] = json.dumps(manifest, ensure_ascii=False, indent=2)
    report = [
        '已将用户提供的两张 LaTeX 表中的 12 个 RBLA+ 值替换为当前工作区最新 RankScale 日志的最高 accuracy × 100。',
        '',
        '先按数据集与秩分配选择文件名时间戳最新的运行，再取该运行已记录轮次中的最高值；没有在旧、新方法或多次运行之间择优。显示保留两位小数。',
        '',
        '10 项日志覆盖 round 0–100。SVHN correct 仅覆盖 0–63，MNLI mismatch 仅覆盖 0–42，按用户确认填入目前最高值并用 † 标记。',
        '',
        'MRPC、QQP、RTE、SST-2 和 MNIST 系列没有找到新的 RankScale 日志，保留用户原表值。这些原值没有在本次检查中重新从原始日志验证。其他方法的所有数值保持不变。更新后重新判断受影响行的最佳结果加粗。',
        '',
        '| 数据集 | 秩分配 | 原 RBLA+ (%) | 新 RBLA+ (%) | 峰值日志 round | 已记录轮次 | 状态 |',
        '| --- | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for r in changes:
        report.append(f"| {r['label']} | {r['assignment']} | {r['original_accuracy']} | {r['display_accuracy']}{'' if r['complete'] else '†'} | {r['peak_round']} | 0–{r['last_round']} | {'完整' if r['complete'] else '未完整'} |")
    report += ['', '来源逐项如下。已核对日志中的 server/client strategy 均为 rbla_plus_rank_scale、聚合器为 sp_plus、数据集及分布、客户端有效 rank ratios 与配置列表一致，并根据样本量与 rank ratio 的相关性验证 correct/mismatch 方向。', '']
    for r in changes:
        report.append(f"- {r['label']} / {r['assignment']}: [{Path(r['path']).name}](../{r['path']})，峰值 round={r['peak_round']}，rank-volume Spearman={r['rank_volume_spearman']:.6f}。")
    report += ['', '两张表保留原宏、列序、字号和分组；表下注明 RankScale 替换范围与两项日志不完整的情况。验证了 12 处 RBLA+ 数值替换、2 处 †、MNIST 系列原文完整保留、其他方法数值不变、LaTeX 花括号平衡。当前环境没有发现 LaTeX 编译器，因此未编译 PDF。', '']
    outputs[str(OUT / 'rbla_plus_tables_update_audit_2026-09-14_zh.md')] = '\n'.join(report)
    # Emit to the workspace file editor, keeping the analysis process read-only.
    print(json.dumps(outputs, ensure_ascii=True))


if __name__ == '__main__':
    main()
