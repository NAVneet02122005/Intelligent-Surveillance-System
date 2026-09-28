def get_class_name(class_id: int) -> str:
    """Mock mapping for COCO dataset classes."""
    classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    return classes.get(class_id, f"unknown_{class_id}")


def is_suspicious(class_name: str) -> bool:
    """Define rules for what triggers a database log and an alert."""
    return class_name in ["person", "car", "motorcycle", "truck"]
