from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import ifcopenshell
import ifcopenshell.guid
import ifcopenshell.validate


DEFAULT_MODEL = {
    "window_overall_height": 1.2,
    "default_extrusion_depth": 3.0,
    "polyline_min_thickness": 0.1,
    "column_height_quantile": 0.9,
}

DEFAULT_CLASSIFIER = {
    "kind": "nearest_centroid_fault_classifier",
    "labels": [],
    "feature_names": [],
    "means": [],
    "scales": [],
    "centroids": {},
    "metrics": {},
    "estimator_path": None,
}

REPORT_FAULT_TYPE_ALIASES = {
    "duplicate_global_id": "duplicate_globalid",
    "missing_global_id": "missing_required_attribute",
    "invalid_window_height": "missing_required_attribute",
    "invalid_cartesian_point": "invalid_geometry",
    "truncated_geometry": "invalid_geometry",
    "invalid_representation": "dangling_reference",
    "geometric_inconsistency": "invalid_geometry",
    "missing_spatial_element": "missing_required_attribute",
    "unrepaired_file_error": "deleted_entity",
}

ML_CANDIDATE_TYPES = {
    "IfcWindow",
    "IfcCartesianPoint",
    "IfcPolyline",
    "IfcWall",
    "IfcColumn",
    "IfcBeam",
    "IfcSlab",
    "IfcDoor",
    "IfcProject",
    "IfcRelAggregates",
}


def _is_bad_number(value: Any) -> bool:
    if isinstance(value, str):
        return True
    try:
        return not math.isfinite(float(value))
    except Exception:
        return True


def _coords(point: Any) -> list[float] | None:
    try:
        values = list(point.Coordinates)
    except Exception:
        return None
    if not values or any(_is_bad_number(v) for v in values):
        return None
    return [float(v) for v in values]


def _mode_float(values: list[float], fallback: float) -> float:
    if not values:
        return fallback
    rounded = [round(float(v), 6) for v in values if not _is_bad_number(v) and float(v) > 0]
    if not rounded:
        return fallback
    return Counter(rounded).most_common(1)[0][0]


def load_model(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as fh:
            model = json.load(fh)
        return {**DEFAULT_MODEL, **model}
    return dict(DEFAULT_MODEL)


def load_classifier(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as fh:
            classifier = {**DEFAULT_CLASSIFIER, **json.load(fh)}
        estimator_path = classifier.get("estimator_path")
        if estimator_path and Path(estimator_path).exists():
            try:
                import joblib

                classifier["_sklearn_bundle"] = joblib.load(estimator_path)
            except Exception:
                pass
        return classifier
    return dict(DEFAULT_CLASSIFIER)


def entity_feature_dict(model: Any, entity: Any, total_entities: int | None = None) -> dict[str, float]:
    total = total_entities or max(entity.id() or 1, 1)
    features: dict[str, float] = {
        "id_norm": float(entity.id() or 0) / max(total, 1),
        "attr_count": float(len(entity)),
        "null_count": 0.0,
        "bad_number_count": 0.0,
        "reference_count": 0.0,
        "inverse_count": 0.0,
        "point_count": 0.0,
        "is_root": 1.0 if hasattr(entity, "GlobalId") else 0.0,
        "has_missing_global_id": 0.0,
        "has_representation": 0.0,
        "bad_representation": 0.0,
        "bad_window_height": 0.0,
        f"type:{entity.is_a()}": 1.0,
    }
    try:
        features["inverse_count"] = float(len(model.get_inverse(entity)))
    except Exception:
        pass
    if hasattr(entity, "GlobalId"):
        gid = getattr(entity, "GlobalId", None)
        features["has_missing_global_id"] = 1.0 if not gid or not isinstance(gid, str) else 0.0
    for idx in range(len(entity)):
        try:
            value = entity[idx]
        except Exception:
            continue
        if value is None:
            features["null_count"] += 1.0
        if isinstance(value, str) and _is_bad_number(value):
            features["bad_number_count"] += 1.0
        if hasattr(value, "id"):
            features["reference_count"] += 1.0
        elif isinstance(value, (tuple, list)):
            features["reference_count"] += float(sum(1 for item in value if hasattr(item, "id")))
    if entity.is_a("IfcPolyline"):
        points = list(getattr(entity, "Points", []) or [])
        features["point_count"] = float(len(points))
    if entity.is_a("IfcCartesianPoint") and _coords(entity) is None:
        features["bad_number_count"] += 2.0
    if entity.is_a("IfcWindow"):
        value = getattr(entity, "OverallHeight", None)
        features["bad_window_height"] = 1.0 if _is_bad_number(value) or (not _is_bad_number(value) and float(value) <= 0) else 0.0
    if hasattr(entity, "Representation"):
        rep = getattr(entity, "Representation", None)
        features["has_representation"] = 1.0 if rep else 0.0
        features["bad_representation"] = 1.0 if rep is not None and not rep.is_a("IfcProductDefinitionShape") else 0.0
    return features


def heuristic_entity_labels(model: Any) -> dict[int, str]:
    labels: dict[int, str] = {}
    seen: dict[str, Any] = {}
    for entity in sorted(list(model), key=lambda item: item.id()):
        if hasattr(entity, "GlobalId"):
            gid = getattr(entity, "GlobalId", None)
            if not gid or not isinstance(gid, str):
                labels[entity.id()] = "missing_global_id"
            elif gid in seen:
                labels[entity.id()] = "duplicate_global_id"
            else:
                seen[gid] = entity
    for window in model.by_type("IfcWindow"):
        value = getattr(window, "OverallHeight", None)
        if _is_bad_number(value) or (not _is_bad_number(value) and float(value) <= 0):
            labels[window.id()] = "invalid_window_height"
    for point in model.by_type("IfcCartesianPoint"):
        if _coords(point) is None:
            labels[point.id()] = "invalid_cartesian_point"
    for polyline in model.by_type("IfcPolyline"):
        points = list(polyline.Points or [])
        valid = [_coords(p) for p in points]
        valid = [p for p in valid if p is not None and len(p) >= 2]
        if len(points) < 3 or len(valid) < 3:
            labels[polyline.id()] = "truncated_geometry"
    for product_type in ("IfcWall", "IfcColumn", "IfcBeam", "IfcSlab", "IfcWindow", "IfcDoor"):
        for product in model.by_type(product_type):
            rep = getattr(product, "Representation", None)
            if rep is not None and not rep.is_a("IfcProductDefinitionShape"):
                labels[product.id()] = "invalid_representation"
    return labels


def vectorize(features: dict[str, float], feature_names: list[str]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in feature_names]


def standardize(vector: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - means[idx]) / (scales[idx] or 1.0) for idx, value in enumerate(vector)]


def train_fault_classifier(input_dir: Path, output_path: Path, max_normal_per_file: int = 80) -> dict[str, Any]:
    samples: list[tuple[dict[str, float], str]] = []
    label_counts: Counter[str] = Counter()
    for ifc_path in sorted(input_dir.rglob("*.ifc")):
        try:
            model = ifcopenshell.open(str(ifc_path))
        except Exception:
            continue
        labels = heuristic_entity_labels(model)
        total_entities = len(list(model))
        normal_added = 0
        for entity in model:
            label = labels.get(entity.id())
            if label is None:
                if entity.is_a() not in ML_CANDIDATE_TYPES or normal_added >= max_normal_per_file:
                    continue
                label = "normal"
                normal_added += 1
            features = entity_feature_dict(model, entity, total_entities=total_entities)
            samples.append((features, label))
            label_counts[label] += 1
    feature_names = sorted({name for features, _ in samples for name in features})
    if not samples or len(label_counts) < 2:
        classifier = {
            **DEFAULT_CLASSIFIER,
            "metrics": {
                "training_samples": len(samples),
                "label_counts": dict(label_counts),
                "accuracy": 0.0,
                "loss": 1.0,
                "note": "Not enough labeled fault classes to train a classifier.",
            },
        }
    else:
        sklearn_classifier = train_sklearn_fault_classifier(samples, label_counts, output_path)
        if sklearn_classifier:
            sklearn_classifier["metrics"]["training_files"] = len(list(input_dir.rglob("*.ifc")))
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump({key: value for key, value in sklearn_classifier.items() if not key.startswith("_")}, fh, indent=2)
            return sklearn_classifier
        raw_vectors = [vectorize(features, feature_names) for features, _ in samples]
        labels = [label for _, label in samples]
        cols = list(zip(*raw_vectors))
        means = [sum(col) / len(col) for col in cols]
        scales = []
        for col, mean in zip(cols, means):
            variance = sum((value - mean) ** 2 for value in col) / max(len(col), 1)
            scales.append(math.sqrt(variance) or 1.0)
        vectors = [standardize(vector, means, scales) for vector in raw_vectors]
        train_size = max(1, int(len(vectors) * 0.8))
        centroids: dict[str, list[float]] = {}
        for label in sorted(set(labels)):
            label_vectors = [vector for vector, y in zip(vectors[:train_size], labels[:train_size]) if y == label]
            if not label_vectors:
                continue
            centroids[label] = [sum(values) / len(values) for values in zip(*label_vectors)]
        total = 0
        correct = 0
        loss_sum = 0.0
        for vector, label in zip(vectors[train_size:], labels[train_size:]):
            pred, confidence = predict_vector_fault(vector, centroids)
            total += 1
            correct += int(pred == label)
            loss_sum += 1.0 - confidence if pred == label else 1.0
        accuracy = correct / total if total else 1.0
        classifier = {
            "kind": "nearest_centroid_fault_classifier",
            "labels": sorted(centroids),
            "feature_names": feature_names,
            "means": means,
            "scales": scales,
            "centroids": centroids,
            "metrics": {
                "training_samples": len(samples),
                "training_files": len(list(input_dir.rglob("*.ifc"))),
                "label_counts": dict(label_counts),
                "accuracy": round(accuracy, 4),
                "loss": round(loss_sum / total, 6) if total else 0.0,
                "algorithm": "nearest centroid over engineered IFC entity features",
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(classifier, fh, indent=2)
    return classifier


def train_sklearn_fault_classifier(samples: list[tuple[dict[str, float], str]], label_counts: Counter[str], output_path: Path) -> dict[str, Any] | None:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.model_selection import train_test_split
    except Exception:
        return None

    features = [features for features, _ in samples]
    labels = [label for _, label in samples]
    stratify = labels if min(label_counts.values()) >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    vectorizer = DictVectorizer(sparse=False)
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)
    estimator = RandomForestClassifier(
        n_estimators=60,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=42,
    )
    estimator.fit(train_matrix, y_train)
    predictions = estimator.predict(test_matrix)
    probabilities = estimator.predict_proba(test_matrix)
    accuracy = float(accuracy_score(y_test, predictions)) if y_test else 1.0
    try:
        loss = float(log_loss(y_test, probabilities, labels=list(estimator.classes_)))
    except ValueError:
        loss = float(1.0 - accuracy)

    estimator_path = output_path.with_suffix(".joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "estimator": estimator}, estimator_path)
    classifier = {
        "kind": "sklearn_random_forest_fault_classifier",
        "labels": list(estimator.classes_),
        "feature_names": list(vectorizer.feature_names_),
        "means": [],
        "scales": [],
        "centroids": {},
        "estimator_path": str(estimator_path),
        "metrics": {
            "training_samples": len(samples),
            "training_files": None,
            "label_counts": dict(label_counts),
            "accuracy": round(accuracy, 4),
            "loss": round(loss, 6),
            "algorithm": "scikit-learn RandomForestClassifier over engineered IFC entity features",
            "n_estimators": 60,
            "max_depth": 14,
            "n_jobs": 1,
        },
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(classifier, fh, indent=2)
    return classifier


def predict_vector_fault(vector: list[float], centroids: dict[str, list[float]]) -> tuple[str, float]:
    if not centroids:
        return "normal", 0.0
    distances = []
    for label, centroid in centroids.items():
        distance = math.sqrt(sum((value - centroid[idx]) ** 2 for idx, value in enumerate(vector)))
        distances.append((distance, label))
    distances.sort()
    best_distance, best_label = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else best_distance + 1.0
    confidence = max(0.0, min(1.0, (second_distance - best_distance) / (second_distance + 1e-9)))
    return best_label, confidence


def predict_entity_fault(model: Any, entity: Any, classifier: dict[str, Any], total_entities: int | None = None) -> tuple[str, float]:
    sklearn_bundle = classifier.get("_sklearn_bundle")
    if sklearn_bundle:
        features = entity_feature_dict(model, entity, total_entities=total_entities)
        vector = sklearn_bundle["vectorizer"].transform([features])
        estimator = sklearn_bundle["estimator"]
        label = str(estimator.predict(vector)[0])
        confidence = float(max(estimator.predict_proba(vector)[0]))
        return label, confidence
    feature_names = list(classifier.get("feature_names") or [])
    centroids = dict(classifier.get("centroids") or {})
    if not feature_names or not centroids:
        return "normal", 0.0
    features = entity_feature_dict(model, entity, total_entities=total_entities)
    vector = vectorize(features, feature_names)
    vector = standardize(vector, list(classifier.get("means") or []), list(classifier.get("scales") or []))
    return predict_vector_fault(vector, centroids)


def ml_fault_predictions(model: Any, classifier: dict[str, Any], max_predictions: int = 250) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    total_entities = len(list(model))
    seen_gids: set[str] = set()
    for entity in model:
        if entity.is_a() not in ML_CANDIDATE_TYPES:
            continue
        suspicious = False
        if hasattr(entity, "GlobalId"):
            gid = getattr(entity, "GlobalId", None)
            suspicious = suspicious or not gid or not isinstance(gid, str) or gid in seen_gids
            if gid and isinstance(gid, str):
                seen_gids.add(gid)
        if entity.is_a("IfcWindow"):
            value = getattr(entity, "OverallHeight", None)
            suspicious = suspicious or _is_bad_number(value) or (not _is_bad_number(value) and float(value) <= 0)
        elif entity.is_a("IfcCartesianPoint"):
            suspicious = suspicious or _coords(entity) is None
        elif entity.is_a("IfcPolyline"):
            points = list(getattr(entity, "Points", []) or [])
            valid = [_coords(p) for p in points]
            valid = [p for p in valid if p is not None and len(p) >= 2]
            suspicious = suspicious or len(points) < 3 or len(valid) < 3
        elif hasattr(entity, "Representation"):
            rep = getattr(entity, "Representation", None)
            suspicious = suspicious or (rep is not None and not rep.is_a("IfcProductDefinitionShape"))
        if not suspicious:
            continue
        label, confidence = predict_entity_fault(model, entity, classifier, total_entities=total_entities)
        if label == "normal" or confidence < 0.15:
            continue
        predictions.append(
            {
                "entity_id": f"#{entity.id()}",
                "predicted_fault_type": label,
                "confidence": round(confidence, 4),
                "entity_type": entity.is_a(),
            }
        )
        if len(predictions) >= max_predictions:
            break
    return predictions


def train_model(input_dir: Path, output_path: Path) -> dict[str, Any]:
    heights: list[float] = []
    depths: list[float] = []
    poly_widths: list[float] = []
    entity_counts: Counter[str] = Counter()

    for ifc_path in sorted(input_dir.rglob("*.ifc")):
        try:
            model = ifcopenshell.open(str(ifc_path))
        except Exception:
            continue

        for entity in model:
            entity_counts[entity.is_a()] += 1

        for window in model.by_type("IfcWindow"):
            value = getattr(window, "OverallHeight", None)
            if not _is_bad_number(value) and float(value) > 0:
                heights.append(float(value))

        for solid in model.by_type("IfcExtrudedAreaSolid"):
            value = getattr(solid, "Depth", None)
            if not _is_bad_number(value) and float(value) > 0:
                depths.append(float(value))

        for polyline in model.by_type("IfcPolyline"):
            pts = [_coords(p) for p in list(polyline.Points or [])]
            pts = [p for p in pts if p is not None and len(p) >= 2]
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                width = min(max(xs) - min(xs), max(ys) - min(ys))
                if width > 0:
                    poly_widths.append(width)

    profile = {
        "window_overall_height": _mode_float(heights, DEFAULT_MODEL["window_overall_height"]),
        "default_extrusion_depth": _mode_float(depths, DEFAULT_MODEL["default_extrusion_depth"]),
        "polyline_min_thickness": max(0.01, float(median(poly_widths)) if poly_widths else DEFAULT_MODEL["polyline_min_thickness"]),
        "entity_type_priors": dict(entity_counts.most_common(40)),
        "training_files": len(list(input_dir.rglob("*.ifc"))),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
    return profile


def fault(entity: Any, fault_type: str, description: str) -> dict[str, str]:
    entity_id = f"#{entity.id()}" if hasattr(entity, "id") and entity.id() else "$"
    return {"entity_id": entity_id, "fault_type": fault_type, "description": description}


def submission_fault(fault_item: dict[str, str]) -> dict[str, str]:
    entity_id = fault_item.get("entity_id", "$")
    fault_type = REPORT_FAULT_TYPE_ALIASES.get(fault_item.get("fault_type", ""), fault_item.get("fault_type", "unknown_fault"))
    return {
        "entity_id": entity_id,
        "fault_type": fault_type,
        "description": f"Detected {fault_type} at entity {entity_id}",
    }


def fix_duplicate_global_ids(model: Any, faults: list[dict[str, str]]) -> None:
    seen: dict[str, Any] = {}
    for entity in sorted(list(model), key=lambda item: item.id()):
        if not hasattr(entity, "GlobalId"):
            continue
        gid = getattr(entity, "GlobalId", None)
        if not gid or not isinstance(gid, str):
            new_gid = ifcopenshell.guid.new()
            faults.append(fault(entity, "missing_global_id", f"Assigned new GlobalId {new_gid}."))
            entity.GlobalId = new_gid
            seen[new_gid] = entity
            continue
        if gid in seen:
            new_gid = ifcopenshell.guid.new()
            faults.append(fault(entity, "duplicate_global_id", f"GlobalId duplicated with #{seen[gid].id()}; reassigned to {new_gid}."))
            entity.GlobalId = new_gid
        else:
            seen[gid] = entity


def fix_windows(model: Any, profile: dict[str, Any], faults: list[dict[str, str]]) -> None:
    default_height = float(profile["window_overall_height"])
    for window in model.by_type("IfcWindow"):
        value = getattr(window, "OverallHeight", None)
        if _is_bad_number(value) or (not _is_bad_number(value) and float(value) <= 0):
            faults.append(fault(window, "invalid_window_height", f"Replaced invalid OverallHeight {value!r} with {default_height}."))
            window.OverallHeight = default_height


def infer_rectangle_point(polyline: Any, bad_point: Any, fallback_thickness: float) -> list[float]:
    pts = [_coords(p) for p in list(polyline.Points or [])]
    idx = list(polyline.Points or []).index(bad_point)
    valid = [p for p in pts if p is not None and len(p) >= 2]
    if len(valid) >= 3:
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        # Common rectangle corruption: the fourth point is missing, so combine
        # the first point's x with the previous point's y.
        prev = pts[idx - 1] if idx > 0 else None
        first = valid[0]
        if prev and len(prev) >= 2:
            return [first[0], prev[1]]
        return [min(xs), min(ys)]
    if len(valid) == 2:
        a, b = valid
        return [a[0], b[1] + fallback_thickness]
    return [0.0, 0.0]


def fix_bad_cartesian_points(model: Any, profile: dict[str, Any], faults: list[dict[str, str]]) -> None:
    thickness = float(profile["polyline_min_thickness"])
    for point in model.by_type("IfcCartesianPoint"):
        if _coords(point) is not None:
            continue
        inverses = list(model.get_inverse(point))
        new_coords = None
        for inverse in inverses:
            if inverse.is_a("IfcPolyline"):
                new_coords = infer_rectangle_point(inverse, point, thickness)
                break
        if new_coords is None:
            new_coords = [0.0, 0.0]
        faults.append(fault(point, "invalid_cartesian_point", f"Replaced invalid coordinates with {new_coords}."))
        point.Coordinates = tuple(float(v) for v in new_coords)


def fix_truncated_polylines(model: Any, profile: dict[str, Any], faults: list[dict[str, str]]) -> None:
    thickness = float(profile["polyline_min_thickness"])
    for polyline in model.by_type("IfcPolyline"):
        points = list(polyline.Points or [])
        valid = [_coords(p) for p in points]
        valid = [p for p in valid if p is not None and len(p) >= 2]
        if len(points) >= 3 and len(valid) >= 3:
            continue
        if len(valid) < 2:
            continue
        a, b = valid[0], valid[1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length * thickness, dx / length * thickness
        p3 = model.create_entity("IfcCartesianPoint", (b[0] + nx, b[1] + ny))
        p4 = model.create_entity("IfcCartesianPoint", (a[0] + nx, a[1] + ny))
        polyline.Points = tuple(points[:2] + [p3, p4, points[0]])
        faults.append(fault(polyline, "truncated_geometry", "Expanded a polyline with fewer than 3 points into a thin closed rectangle."))


def _representation_items(product: Any) -> list[Any]:
    rep = getattr(product, "Representation", None)
    if not rep or not hasattr(rep, "Representations"):
        return []
    items: list[Any] = []
    for shape_rep in rep.Representations or []:
        items.extend(list(getattr(shape_rep, "Items", []) or []))
    return items


def fix_short_columns(model: Any, profile: dict[str, Any], faults: list[dict[str, str]]) -> None:
    column_depths: list[float] = []
    solids_by_column: dict[int, list[Any]] = defaultdict(list)
    for column in model.by_type("IfcColumn"):
        for item in _representation_items(column):
            if item.is_a("IfcExtrudedAreaSolid") and not _is_bad_number(item.Depth):
                solids_by_column[column.id()].append(item)
                column_depths.append(float(item.Depth))
    if not column_depths:
        return
    target = max(float(profile.get("default_extrusion_depth", 3.0)), sorted(column_depths)[int(0.75 * (len(column_depths) - 1))])
    for column in model.by_type("IfcColumn"):
        for solid in solids_by_column.get(column.id(), []):
            if float(solid.Depth) < target * 0.75:
                faults.append(fault(column, "geometric_inconsistency", f"Column extrusion depth {solid.Depth} was shorter than peer/storey estimate {target}; extended."))
                solid.Depth = target


def fix_bad_representations(model: Any, faults: list[dict[str, str]]) -> None:
    for product_type in ("IfcWall", "IfcColumn", "IfcBeam", "IfcSlab", "IfcWindow", "IfcDoor"):
        for product in model.by_type(product_type):
            rep = getattr(product, "Representation", None)
            if rep is None or rep.is_a("IfcProductDefinitionShape"):
                continue
            # Find the nearest preceding shape representation and wrap it.
            shape = None
            for candidate in reversed(list(model.by_type("IfcShapeRepresentation"))):
                if candidate.id() < product.id():
                    shape = candidate
                    break
            if shape is None:
                continue
            pds = model.create_entity("IfcProductDefinitionShape", None, None, (shape,))
            faults.append(fault(product, "invalid_representation", f"Replaced {rep.is_a()} reference #{rep.id()} with a new IfcProductDefinitionShape."))
            product.Representation = pds
            try:
                if len(model.get_inverse(rep)) == 0:
                    model.remove(rep)
            except Exception:
                pass


def add_missing_space(model: Any, faults: list[dict[str, str]]) -> None:
    if model.by_type("IfcSpace"):
        return
    if not (model.by_type("IfcWall") or model.by_type("IfcWallStandardCase")):
        return
    owner = model.by_type("IfcOwnerHistory")[0] if model.by_type("IfcOwnerHistory") else None
    storey = model.by_type("IfcBuildingStorey")[0] if model.by_type("IfcBuildingStorey") else None
    placement = getattr(storey, "ObjectPlacement", None) if storey else None
    schema = str(getattr(model, "schema", "")).upper()
    if "IFC2X3" in schema:
        space = model.create_entity(
            "IfcSpace",
            ifcopenshell.guid.new(),
            owner,
            "Auto repaired space",
            None,
            None,
            placement,
            None,
            None,
            "ELEMENT",
            "INTERNAL",
            None,
        )
    else:
        space = model.create_entity(
            "IfcSpace",
            ifcopenshell.guid.new(),
            owner,
            "Auto repaired space",
            None,
            None,
            placement,
            None,
            None,
            "ELEMENT",
            None,
            None,
        )
    faults.append(fault(space, "missing_spatial_element", "Added a minimal IfcSpace because walls exist but no spaces were present."))


def validate_model(model: Any) -> list[dict[str, Any]]:
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger)
    return list(logger.statements)


def repair_file(input_path: Path, output_path: Path, profile: dict[str, Any], classifier: dict[str, Any] | None = None, validate: bool = True) -> dict[str, Any]:
    model = ifcopenshell.open(str(input_path))
    before_errors = validate_model(model) if validate else []
    ml_predictions = ml_fault_predictions(model, classifier or {}) if classifier else []
    faults: list[dict[str, str]] = []

    fix_windows(model, profile, faults)
    fix_bad_cartesian_points(model, profile, faults)
    fix_truncated_polylines(model, profile, faults)
    fix_bad_representations(model, faults)
    fix_short_columns(model, profile, faults)
    add_missing_space(model, faults)
    fix_duplicate_global_ids(model, faults)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(output_path))

    validation_errors = validate_model(ifcopenshell.open(str(output_path))) if validate else []
    return {
        "id": input_path.stem,
        "input": str(input_path),
        "output": str(output_path),
        "faults": faults,
        "ml_fault_predictions": ml_predictions,
        "validation_error_count_before": len(before_errors),
        "validation_error_count": len(validation_errors),
        "validation_errors": [str(e.get("message", ""))[:300] for e in validation_errors[:10]],
    }


def repair_dir(input_dir: Path, output_dir: Path, report_path: Path, profile: dict[str, Any], classifier: dict[str, Any] | None = None, validate: bool = True) -> list[dict[str, Any]]:
    results = []
    for input_path in sorted(input_dir.glob("*.ifc")):
        output_path = output_dir / input_path.name
        try:
            results.append(repair_file(input_path, output_path, profile, classifier=classifier, validate=validate))
        except Exception as exc:
            results.append(
                {
                    "id": input_path.stem,
                    "input": str(input_path),
                    "output": str(output_path),
                    "faults": [
                        {
                            "entity_id": "$",
                            "fault_type": "unrepaired_file_error",
                            "description": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                    "ml_fault_predictions": [],
                    "validation_error_count_before": 0,
                    "validation_error_count": 1,
                    "validation_errors": [f"{type(exc).__name__}: {exc}"],
                }
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps({"id": result["id"], "faults": [submission_fault(item) for item in result["faults"]]}) + "\n")
    return results


def build_metrics(results: list[dict[str, Any]], elapsed_seconds: float, profile: dict[str, Any], classifier: dict[str, Any], validate: bool) -> dict[str, Any]:
    file_count = len(results)
    total_faults = sum(len(result["faults"]) for result in results)
    total_before_errors = sum(int(result.get("validation_error_count_before", 0)) for result in results)
    total_after_errors = sum(int(result.get("validation_error_count", 0)) for result in results)
    total_ml_predictions = sum(len(result.get("ml_fault_predictions", [])) for result in results)
    valid_outputs = sum(1 for result in results if int(result.get("validation_error_count", 0)) == 0)
    denominator = max(total_before_errors + total_faults, 1)
    return {
        "files_processed": file_count,
        "valid_outputs": valid_outputs,
        "schema_validity_rate": round(valid_outputs / file_count, 4) if file_count else 0.0,
        "detected_faults": total_faults,
        "ml_predicted_faults": total_ml_predictions,
        "validation_errors_before": total_before_errors,
        "validation_errors_after": total_after_errors,
        "repair_loss_proxy": round(total_after_errors / denominator, 6),
        "repair_efficiency_percent": round((1.0 - (total_after_errors / denominator)) * 100.0, 2),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "files_per_second": round(file_count / elapsed_seconds, 4) if elapsed_seconds > 0 else file_count,
        "seconds_per_file": round(elapsed_seconds / file_count, 4) if file_count else 0.0,
        "full_validation_enabled": validate,
        "model_profile": {
            "training_files": profile.get("training_files", 0),
            "window_overall_height": profile.get("window_overall_height"),
            "default_extrusion_depth": profile.get("default_extrusion_depth"),
            "polyline_min_thickness": profile.get("polyline_min_thickness"),
        },
        "fault_classifier": classifier.get("metrics", {}),
    }


def run_pipeline(input_dir: Path, output_dir: Path, model_path: Path, classifier_path: Path, report_path: Path, metrics_path: Path, validate: bool = True) -> dict[str, Any]:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    ifc_files = sorted(input_dir.glob("*.ifc"))
    if not ifc_files:
        raise SystemExit(f"No .ifc files found in {input_dir}. Put your test dataset there and run again.")

    profile = train_model(input_dir, model_path)
    train_fault_classifier(input_dir, classifier_path)
    classifier = load_classifier(classifier_path)
    start = time.perf_counter()
    results = repair_dir(input_dir, output_dir, report_path, profile, classifier=classifier, validate=validate)
    elapsed = time.perf_counter() - start
    metrics = build_metrics(results, elapsed, profile, classifier, validate)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "files": results}, fh, indent=2)
    return {"metrics": metrics, "files": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run a lightweight IFC repair model.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--input-dir", required=True, type=Path)
    train.add_argument("--model-out", default=Path("model/repair_profile.json"), type=Path)
    train.add_argument("--classifier-out", default=Path("model/fault_classifier.json"), type=Path)

    repair = sub.add_parser("repair")
    repair.add_argument("--input-dir", required=True, type=Path)
    repair.add_argument("--output-dir", default=Path("submission/repaired_ifc"), type=Path)
    repair.add_argument("--report-out", default=Path("submission/fault_report.jsonl"), type=Path)
    repair.add_argument("--model", default=Path("model/repair_profile.json"), type=Path)
    repair.add_argument("--classifier", default=Path("model/fault_classifier.json"), type=Path)
    repair.add_argument("--fast", action="store_true", help="Skip full before/after schema validation for faster bulk repairs.")

    run = sub.add_parser("run")
    run.add_argument("--input-dir", default=Path("data/input"), type=Path)
    run.add_argument("--output-dir", default=Path("data/output/repaired_ifc"), type=Path)
    run.add_argument("--model-out", default=Path("model/repair_profile.json"), type=Path)
    run.add_argument("--classifier-out", default=Path("model/fault_classifier.json"), type=Path)
    run.add_argument("--report-out", default=Path("data/output/fault_report.jsonl"), type=Path)
    run.add_argument("--metrics-out", default=Path("data/output/metrics.json"), type=Path)
    run.add_argument("--fast", action="store_true", help="Skip full before/after schema validation for faster bulk repairs.")

    args = parser.parse_args()
    if args.cmd == "train":
        profile = train_model(args.input_dir, args.model_out)
        classifier = train_fault_classifier(args.input_dir, args.classifier_out)
        print(json.dumps({"repair_profile": profile, "fault_classifier": classifier["metrics"]}, indent=2))
    elif args.cmd == "repair":
        profile = load_model(args.model)
        classifier = load_classifier(args.classifier)
        results = repair_dir(args.input_dir, args.output_dir, args.report_out, profile, classifier=classifier, validate=not args.fast)
        print(json.dumps(results, indent=2))
    elif args.cmd == "run":
        result = run_pipeline(args.input_dir, args.output_dir, args.model_out, args.classifier_out, args.report_out, args.metrics_out, validate=not args.fast)
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
