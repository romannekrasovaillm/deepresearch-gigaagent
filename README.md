# Deep Research GigaAgent

Обучение агента для научного синтеза на локальной базе статей с Code Reasoning.

## Обзор

Этот проект реализует полный пайплайн обучения исследовательского агента:

1. **Конвертация библиотеки** — DOCX → структурированный Markdown
2. **Генерация датасета** — синтетические задачи разной сложности
3. **SFT обучение** — обучение формату вызова инструментов
4. **RL обучение** — MOA-style многоцелевая оптимизация

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                      Deep Research Agent                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Bash Tools  │  │   Code      │  │    Scratchpad           │ │
│  │ grep, find  │  │ Interpreter │  │    (Notes)              │ │
│  │ read, verify│  │ Python/JS   │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     Paper Library                               │
│  /mnt/library/                                                  │
│  ├── index.json                                                 │
│  └── by_id/paper_001/                                          │
│      ├── metadata.json                                          │
│      ├── full_text.md                                           │
│      ├── abstract.txt                                           │
│      └── sections/{intro,methods,results}.txt                   │
└─────────────────────────────────────────────────────────────────┘
```

## Быстрый старт

### 1. Установка

```bash
# Клонирование репозитория
git clone <repo-url>
cd deepresearch-gigaagent

# Установка зависимостей
pip install -r requirements.txt
sudo apt install pandoc ripgrep

# Установка training библиотек
git clone https://github.com/PrimeIntellect-ai/prime-rl.git external/prime-rl
git clone https://github.com/PrimeIntellect-ai/verifiers.git external/verifiers
pip install -e ./external/prime-rl -e ./external/verifiers
```

### 2. Подготовка данных

Положите DOCX файлы со статьями в `data/raw_docx/`:

```bash
mkdir -p data/raw_docx
cp /path/to/your/papers/*.docx data/raw_docx/
```

### 3. Запуск пайплайна

```bash
# Полный пайплайн
./run_pipeline.sh --stage all

# Или по этапам:
./run_pipeline.sh --stage library   # Конвертация библиотеки
./run_pipeline.sh --stage data      # Генерация датасета
./run_pipeline.sh --stage sft       # SFT обучение
./run_pipeline.sh --stage rl        # RL обучение
```

## Структура проекта

```
deepresearch-gigaagent/
├── run_pipeline.sh          # Главный скрипт запуска
├── configs/
│   └── default.yaml         # Конфигурация по умолчанию
├── src/
│   ├── library/             # Конвертация DOCX → Markdown
│   │   ├── converter.py     # Конвертер документов
│   │   └── indexer.py       # Индексация библиотеки
│   ├── tools/               # Инструменты агента
│   │   ├── bash_tools.py    # grep, find, read, verify
│   │   ├── code_interpreter.py  # Python/JS песочница
│   │   └── scratchpad.py    # Записная книжка
│   ├── rewards/             # Система наград
│   │   ├── rewards.py       # MOA-style функции награды
│   │   └── judge.py         # LLM-as-Judge
│   ├── data_generation/     # Генерация датасета
│   │   ├── prompts.py       # Промпты для генерации
│   │   └── generator.py     # Генератор задач
│   └── training/            # Обучение
│       ├── environment.py   # RL среда
│       ├── sft_trainer.py   # SFT обучение
│       └── rl_trainer.py    # RL обучение
├── data/
│   ├── raw_docx/           # Исходные DOCX файлы
│   ├── library/            # Структурированная библиотека
│   ├── datasets/           # Сгенерированные датасеты
│   └── workspace/          # Рабочая директория агента
└── outputs/
    ├── sft/                # Чекпоинты SFT
    └── rl/                 # Чекпоинты RL
```

## Инструменты агента

### Bash Tools (детерминированный поиск)

| Инструмент | Описание |
|------------|----------|
| `grep_search` | Поиск по паттерну в библиотеке (ripgrep) |
| `read_file_chunk` | Чтение части файла по строкам |
| `find_files` | Поиск файлов по имени |
| `verify_quote` | Проверка существования цитаты |
| `list_papers` | Список статей из индекса |

### Code Interpreter (аналитический reasoning)

```python
# Агент может выполнять:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
```

**Возможности:**
- Статистический анализ метрик из статей
- Сравнение результатов между papers
- Визуализация трендов
- Математические вычисления

### Scratchpad (рабочие заметки)

```python
add_to_notes(text, tag="finding")  # Сохранить находку
read_notes()                        # Прочитать все заметки
```

## Система наград (MOA-style)

| Награда | Вес | Что измеряет |
|---------|-----|--------------|
| `R_fact` | 0.4 | Фактологическая точность (LLM-Judge) |
| `R_process` | 0.2 | Эффективность поиска |
| `R_citation` | 0.2 | Точность ссылок (программная проверка) |
| `R_code` | 0.2 | Качество code reasoning |

## Curriculum Learning

Обучение происходит по уровням сложности:

| Уровень | Тип задач | Инструменты |
|---------|-----------|-------------|
| 1 | Retrieval | grep, find, list_papers |
| 2 | Extraction | + add_to_notes |
| 3 | Compute | + execute_python |
| 4 | Multi-hop | + verify_quote |
| 5 | Visualize | + save_figure |
| 6 | Synthesis | все инструменты |

## Конфигурация

Основные параметры в `configs/default.yaml`:

```yaml
# Обучение
sft:
  learning_rate: 2.0e-5
  num_epochs: 3

rl:
  algorithm: "ppo"
  total_steps: 10000
  curriculum:
    enabled: true
    levels: 6

# Награды
rewards:
  weights:
    R_fact: 0.4
    R_process: 0.2
    R_citation: 0.2
    R_code: 0.2
```

## Расчёт ресурсов (4×H200)

Для модели GigaChat Lightning (10B параметров) в BF16:

| Компонент | VRAM |
|-----------|------|
| Веса модели | ~20 ГБ |
| Оптимизатор (AdamW) | ~40-60 ГБ |
| Градиенты | ~20 ГБ |
| **Итого** | ~80-100 ГБ |

С FSDP на 4 картах: ~25-30 ГБ на карту.

## Добавление новых статей

```bash
# Добавить одну статью
python -c "
from src.library.converter import add_paper
add_paper('path/to/new_paper.docx', 'data/library')
"

# Или положить в raw_docx и перезапустить:
./run_pipeline.sh --stage library --skip-deps
```

Новые статьи доступны агенту сразу без переобучения!

## Примеры использования

### Простой поиск

```
Q: В какой статье описан метод PPO-Clip?
A: Метод PPO-Clip описан в paper_042 (Schulman et al., 2017)...
```

### Сравнение методов

```
Q: Сравни подходы к reward shaping в статьях A и B
A: В статье A используется intrinsic motivation на основе...,
   тогда как B применяет learned reward model...
```

### Вычисления

```
Q: Посчитай средний прирост accuracy по 5 статьям
A: [Агент выполняет Python код]
   Средний прирост составляет 12.3% (±2.1% std dev)
```

## Лицензия

MIT

## Авторы

Deep Research Agent Training Pipeline
