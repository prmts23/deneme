# ABCD Harmonic Pattern Detector

Professional deep learning system for detecting ABCD harmonic patterns in financial markets using Convolutional Neural Networks (CNN).

## 🎯 Features

- **Synthetic Pattern Generation**: Generate realistic ABCD patterns with configurable parameters
- **Automated Dataset Creation**: Create labeled training datasets from synthetic patterns
- **CNN Architecture**: Custom CNN model with batch normalization and dropout
- **Transfer Learning**: Support for pre-trained models (EfficientNet, ResNet, etc.)
- **Real-time Detection**: Detect patterns in live OHLCV data streams
- **Comprehensive Backtesting**: Full backtesting engine with performance metrics
- **Risk Management**: Built-in stop loss, take profit, and position sizing
- **Extensible Configuration**: Easily configurable via dataclasses

## 📁 Project Structure

```
deneme/
├── abcd_pattern/           # Core pattern detection package
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── pattern_generator.py  # ABCD pattern generation
│   ├── data_generator.py  # Training data creation
│   ├── model.py           # CNN model architecture
│   ├── trainer.py         # Training pipeline
│   ├── detector.py        # Real-time pattern detection
│   └── utils.py           # Utility functions
├── backtest/              # Backtesting engine
│   ├── __init__.py
│   └── abcd_backtest.py   # Backtest implementation
├── scripts/               # Executable scripts
│   ├── generate_dataset.py
│   ├── train.py
│   └── run_backtest.py
├── models/                # Saved models
├── data_abcd/            # Training datasets
├── logs/                 # Training logs
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd deneme

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Dataset

```python
from abcd_pattern import DatasetGenerator, setup_logging

# Setup logging
setup_logging(log_level="INFO")

# Create generator
generator = DatasetGenerator()

# Generate dataset (1000 positive + 1000 negative samples)
generator.generate_dataset(
    n_positive=1000,
    n_negative=1000,
    output_dir="data_abcd"
)
```

### 3. Train Model

```python
from abcd_pattern import Trainer, setup_logging

# Setup logging
setup_logging(log_level="INFO")

# Create trainer
trainer = Trainer()

# Load datasets
train_ds, val_ds = trainer.load_dataset("data_abcd")

# Train model
history = trainer.train(
    train_ds,
    val_ds,
    model_save_path="models/abcd_detector.h5",
    log_dir="logs"
)

# Plot training history
trainer.plot_training_history("training_history.png")
```

### 4. Detect Patterns

```python
from abcd_pattern import ABCDDetector, load_ohlcv_data

# Load trained model
detector = ABCDDetector(model_path="models/abcd_detector.h5")

# Load OHLCV data
df = load_ohlcv_data("your_data.csv")

# Detect pattern in latest 20 bars
detection = detector.detect_latest(df, lookback=20)

if detection:
    print(f"Pattern detected with {detection['confidence']:.2%} confidence")
    print(f"Price: {detection['price']}")
```

### 5. Run Backtest

```python
from abcd_pattern import ABCDDetector, load_ohlcv_data, BacktestConfig
from backtest import ABCDBacktest

# Load model and data
detector = ABCDDetector("models/abcd_detector.h5")
df = load_ohlcv_data("historical_data.csv")

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    position_size_pct=0.1,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    detection_threshold=0.7
)

# Run backtest
backtest = ABCDBacktest(detector, config)
result = backtest.run(df, lookback=20, check_interval=5)

# Print results
backtest.print_results(result)

# Plot results
backtest.plot_results(result, save_path="backtest_results.png")
```

## 🔧 Configuration

All configuration is managed through dataclasses in `abcd_pattern/config.py`:

### Pattern Configuration

```python
from abcd_pattern import PatternConfig

config = PatternConfig(
    min_bars=7,
    max_bars=30,
    noise_level=0.01,
    min_retracement=0.382,  # 38.2% Fibonacci
    max_retracement=0.886,  # 88.6% Fibonacci
    min_abcd_ratio=0.9,
    max_abcd_ratio=1.618
)
```

### Model Configuration

```python
from abcd_pattern import ModelConfig

config = ModelConfig(
    input_shape=(224, 224, 3),
    conv_filters=(32, 64, 128, 256),
    dense_units=(256, 128),
    dropout_rate=0.5,
    learning_rate=1e-4,
    batch_size=32,
    epochs=50
)
```

### Backtest Configuration

```python
from abcd_pattern import BacktestConfig

config = BacktestConfig(
    initial_capital=10000.0,
    position_size_pct=0.1,      # 10% per trade
    max_positions=3,
    stop_loss_pct=0.02,         # 2% stop loss
    take_profit_pct=0.04,       # 4% take profit
    detection_threshold=0.7,    # 70% confidence
    commission_pct=0.001        # 0.1% commission
)
```

## 📊 ABCD Pattern Basics

The ABCD pattern is a harmonic pattern that consists of three price swings:

- **A to B**: Initial price movement
- **B to C**: Retracement (typically 38.2% - 88.6% Fibonacci)
- **C to D**: Extension (CD/AB ratio typically 0.9 - 1.618)

### Bullish ABCD
```
Price
  C
  ↗
B   ↘
      D
A
```

### Bearish ABCD
```
Price
B   ↗
  ↘   D
  A C
```

## 📈 Performance Metrics

The backtest engine calculates comprehensive metrics:

- **Win Rate**: Percentage of profitable trades
- **Total PnL**: Absolute profit/loss
- **Return %**: Percentage return on capital
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Mean time in trades

## 🛠️ Advanced Usage

### Custom Model Architecture

```python
from abcd_pattern import ABCDPatternCNN, ModelConfig

# Configure custom architecture
config = ModelConfig(
    conv_filters=(64, 128, 256, 512),
    dense_units=(512, 256, 128),
    dropout_rate=0.6
)

# Build model
model_builder = ABCDPatternCNN(config)
model = model_builder.build_model()
```

### Transfer Learning

```python
model_builder = ABCDPatternCNN(config)
model = model_builder.build_transfer_learning_model(
    base_model_name="EfficientNetB0",
    trainable_layers=20
)
```

### Scanning for Patterns

```python
detector = ABCDDetector("models/abcd_detector.h5")

# Scan entire dataset with sliding window
detections = detector.scan_dataframe(
    df,
    window_size=20,
    step=5
)

# Filter overlapping detections
filtered = detector.filter_overlapping_detections(
    detections,
    min_separation=10
)

# Get statistics
stats = detector.get_pattern_statistics(filtered)
print(f"Found {stats['count']} patterns")
print(f"Average confidence: {stats['avg_confidence']:.2%}")
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=abcd_pattern tests/
```

## 📝 TODO

- [ ] Add more harmonic patterns (Gartley, Butterfly, Bat)
- [ ] Implement data augmentation
- [ ] Add hyperparameter tuning
- [ ] Create web dashboard for monitoring
- [ ] Add support for multiple timeframes
- [ ] Implement ensemble models
- [ ] Add real-time trading integration

## 📄 License

MIT License

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 References

- [Harmonic Trading Volume One](https://harmonictrader.com/)
- [Scott M. Carney - Harmonic Trading](https://harmonictrader.com/harnessing-the-power-of-harmonic-patterns/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

**Disclaimer**: This software is for educational purposes only. Use at your own risk. Past performance does not guarantee future results.
