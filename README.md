# Deep Research Agent

Научный синтез на локальной базе статей с Code Reasoning.

Агент для исследовательских задач, обученный через Full RL на локальной библиотеке научных статей. Использует детерминированные bash-инструменты для навигации по файлам и Python-интерпретатор для аналитического reasoning.

## Особенности

- **Full RL обучение** на 4×H200 без LoRA (10B параметров модели)
- **Гибридные инструменты**: bash для поиска + Python для анализа
- **Code Reasoning**: статистический анализ, вычисления, визуализация
- **MOA-style награды**: R_fact, R_process, R_citation, R_code
- **Curriculum Learning**: от простого поиска к полному синтезу
- **Инкрементальная база**: добавление новых статей без переобучения

## Структура проекта

```
deepresearch-gigaagent/
├── run_pipeline.sh          # Единый скрипт запуска всего пайплайна
├── configs/
│   ├── training.yaml        # Конфигурация обучения
│   └── reward.yaml          # Настройки функций наград
├── src/
│   ├── library/             # Конвертация DOCX → Markdown
│   ├── tools/               # Инструменты агента
│   │   ├── bash_tools.py    # grep_search, read_file_chunk, etc.
│   │   ├── code_interpreter.py  # execute_python
│   │   └── scratchpad.py    # add_to_notes, read_notes
│   ├── environment/         # RL среда
│   ├── rewards/             # MOA-style награды
│   ├── data/                # Генерация синтетических данных
│   └── training/            # SFT и RL обучение
├── scripts/                 # CLI-скрипты
└── data/                    # Данные (создаются при запуске)
    ├── raw_docx/            # Исходные DOCX-файлы
    ├── library/             # Структурированная библиотека
    ├── datasets/            # Обучающие данные
    └── workspace/           # Рабочая директория агента
```

## Быстрый старт

### 1. Подготовка

```bash
# Клонирование
git clone <repo_url>
cd deepresearch-gigaagent

# Добавьте DOCX-файлы статей
mkdir -p data/raw_docx
cp /path/to/papers/*.docx data/raw_docx/

# Установите API ключи (для генерации данных)
export OPENAI_API_KEY="sk-..."  # или ANTHROPIC_API_KEY
```

### 2. Запуск полного пайплайна

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh all
```

### 3. Запуск отдельных этапов

```bash
# Только установка зависимостей
./run_pipeline.sh setup

# Конвертация статей
./run_pipeline.sh convert

# Генерация обучающих данных
./run_pipeline.sh generate

# SFT обучение
./run_pipeline.sh sft

# RL обучение
./run_pipeline.sh rl

# Оценка модели
./run_pipeline.sh eval
```

## Конфигурация

### Переменные окружения

```bash
# Пути
export DOCX_DIR="./data/raw_docx"
export LIBRARY_DIR="./data/library"
export DATASET_DIR="./data/datasets"

# Модель
export MODEL_NAME="GigaChat-Lightning-Instruct"
export MODEL_PATH="/path/to/local/model"  # опционально

# Обучение
export NUM_GPUS=4
export NUM_SAMPLES=10000
export SFT_EPOCHS=3
export RL_STEPS=10000
export RL_ALGORITHM="grpo"  # или "ppo"

# API
export GENERATOR_MODEL="gpt-4o"
export JUDGE_MODEL="deepseek-v3"
```

### Конфигурационные файлы

- `configs/training.yaml` - параметры обучения, FSDP, curriculum
- `configs/reward.yaml` - веса и параметры функций наград

## Инструменты агента

### Bash-инструменты (навигация)

```python
grep_search(pattern, path=None)      # Поиск по библиотеке
read_file_chunk(path, start, num)    # Чтение файла
find_files(name_pattern)             # Поиск файлов
list_papers(topic=None)              # Список статей
verify_quote(path, snippet)          # Проверка цитаты
```

### Code Interpreter (анализ)

```python
execute_python(code)  # Выполнение Python-кода
# Доступно: pandas, numpy, scipy, matplotlib, sympy
```

### Scratchpad (память)

```python
add_to_notes(text, section=None)  # Сохранение находок
read_notes()                       # Чтение записей
save_figure(path, name=None)       # Сохранение графиков
```

## Система наград

| Награда | Вес | Что измеряет |
|---------|-----|--------------|
| R_fact | 0.40 | Фактологическая точность (LLM-Judge) |
| R_process | 0.20 | Эффективность поиска |
| R_citation | 0.20 | Точность цитирования |
| R_code | 0.20 | Качество code reasoning |

## Требования

### Hardware

- 4× NVIDIA H200 (141 ГБ каждая) - для Full RL
- Или меньше GPU с LoRA-адаптерами

### Software

- Python 3.10+
- PyTorch 2.1+
- pandoc (для конвертации DOCX)
- CUDA 12.0+

### Зависимости

```bash
pip install -e ".[training,dev]"

# Для распределённого RL
pip install -e "git+https://github.com/PrimeIntellect-ai/prime-rl.git"
pip install -e "git+https://github.com/PrimeIntellect-ai/verifiers.git"
```

## Расширенное использование

### Добавление новых статей

```bash
# Добавить одну статью
python scripts/convert_library.py add paper.docx data/library

# Новые статьи сразу доступны агенту без переобучения
```

### Предпросмотр данных

```bash
# Посмотреть примеры генерируемых задач
python scripts/generate_dataset.py preview data/library --type multihop
```

### Распределённое RL обучение

Для полноценного обучения на 4×H200:

```bash
torchrun --nproc_per_node=4 \
    -m prime_rl.train \
    --config checkpoints/rl/prime_rl_config.yaml
```

## Curriculum Learning

Обучение проходит через этапы возрастающей сложности:

1. **Retrieval** (2000 шагов): простой поиск фактов
2. **Reasoning** (3000 шагов): сравнение + вычисления
3. **Synthesis** (5000 шагов): полный синтез с визуализацией

## Примеры использования агента

### Поиск информации

```
Q: Какой размер батча использовался в эксперименте PPO-Clip?
A: [Агент ищет в библиотеке, находит paper_042, читает methods.txt]
   Размер батча составлял 256 согласно разделу Methods статьи paper_042.
```

### Сравнительный анализ

```
Q: Сравни подходы к reward shaping в статьях A и B
A: [Агент читает обе статьи, сохраняет заметки, анализирует]
   В статье A используется dense reward с формированием...
   В статье B применяется sparse reward с...
```

### Вычислительный анализ

```
Q: Посчитай средний прирост accuracy на GSM8K по 5 статьям
A: [Агент извлекает числа, выполняет Python-код]
   import pandas as pd
   df = pd.DataFrame({'paper': [...], 'accuracy': [...]})
   Средний прирост: 12.3%
```

## Лицензия

MIT

## Ссылки

- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) - Distributed RL training
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) - Tool environments for agents
