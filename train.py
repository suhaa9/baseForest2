from pathlib import Path
import argparse
from typing import Optional

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, log_loss, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import SatelliteDataset
from model import MLPConfig, build_classifier


def resolve_data_dir(data_dir: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    data_candidate = Path(data_dir).expanduser()
    if not data_candidate.is_absolute():
        data_candidate = (Path.cwd() / data_candidate).resolve()

    if data_candidate.exists():
        return data_candidate

    alt_candidate = (script_dir / Path(data_dir)).resolve()
    if alt_candidate.exists():
        return alt_candidate

    raise FileNotFoundError(
        "Dataset directory not found. Checked:\n"
        f"- {data_candidate}\n"
        f"- {alt_candidate}"
    )


def prepare_split(base_path: Path, split: str, image_size) -> Optional[SatelliteDataset]:
    try:
        return SatelliteDataset(str(base_path), split=split, image_size=image_size)
    except (FileNotFoundError, RuntimeError):
        return None


def main(data_dir: str, epochs: int = 5, batch_size: int = 128, save_path: str = 'model.joblib'):
    data_path = resolve_data_dir(data_dir)

    save_path = Path(save_path)
    if not save_path.is_absolute():
        save_path = (Path(__file__).resolve().parent / save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = SatelliteDataset(str(data_path), split='train')

    val_ds = prepare_split(data_path, 'val', train_ds.image_size)
    test_ds = prepare_split(data_path, 'test', train_ds.image_size)

    train_X, train_y_labels = train_ds.to_arrays()
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)

    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y_labels)
    num_classes = len(label_encoder.classes_)

    def transform_split(ds: Optional[SatelliteDataset]):
        if not ds:
            return None, None
        features, labels = ds.to_arrays()
        return scaler.transform(features), label_encoder.transform(labels)

    val_X, val_y = transform_split(val_ds)
    test_X, test_y = transform_split(test_ds)

    config = MLPConfig(batch_size=batch_size)
    model = build_classifier(train_X.shape[1], num_classes, config=config)

    classes = np.arange(num_classes)
    for epoch in range(epochs):
        model.partial_fit(train_X, train_y, classes=classes)
        train_probs = model.predict_proba(train_X)
        train_preds = np.argmax(train_probs, axis=1)
        train_loss = log_loss(train_y, train_probs)
        train_acc = accuracy_score(train_y, train_preds)
        train_prec = precision_score(train_y, train_preds, average='weighted', zero_division=0)
        msg = f"Epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_prec={train_prec:.4f}"
        if val_X is not None and val_y is not None:
            val_probs = model.predict_proba(val_X)
            val_preds = np.argmax(val_probs, axis=1)
            val_loss = log_loss(val_y, val_probs)
            val_acc = accuracy_score(val_y, val_preds)
            val_prec = precision_score(val_y, val_preds, average='weighted', zero_division=0)
            msg += f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_prec={val_prec:.4f}"
        print(msg)

    metrics = {
        'train': {
            'accuracy': float(accuracy_score(train_y, np.argmax(model.predict_proba(train_X), axis=1))),
            'precision': float(precision_score(train_y, np.argmax(model.predict_proba(train_X), axis=1), average='weighted', zero_division=0)),
        }
    }

    if val_X is not None and val_y is not None:
        val_preds = np.argmax(model.predict_proba(val_X), axis=1)
        metrics['val'] = {
            'accuracy': float(accuracy_score(val_y, val_preds)),
            'precision': float(precision_score(val_y, val_preds, average='weighted', zero_division=0)),
        }

    if test_X is not None and test_y is not None:
        test_probs = model.predict_proba(test_X)
        test_preds = np.argmax(test_probs, axis=1)
        test_loss = log_loss(test_y, test_probs)
        test_acc = accuracy_score(test_y, test_preds)
        test_prec = precision_score(test_y, test_preds, average='weighted', zero_division=0)
        metrics['test'] = {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
        }
        print(f"Test loss={test_loss:.4f} acc={test_acc:.4f} prec={test_prec:.4f}")

    dump({
        'model': model,
        'classes': label_encoder.classes_,
        'scaler': scaler,
        'image_size': train_ds.image_size,
        'metrics': metrics,
    }, str(save_path))
    print(f'Saved model to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', help='Path to data folder')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=128, help='Mini-batch size used by the MLPClassifier')
    parser.add_argument('--out', default='model.joblib', help='Output path for the trained model bundle')
    args = parser.parse_args()
    main(args.data, args.epochs, args.batch, args.out)
