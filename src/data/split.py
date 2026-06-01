from sklearn.model_selection import train_test_split


def make_train_val_test_split(
    X,
    y,
    test_size=0.2,
    validation_size=0.25,
    random_state=42,
):
    """Create a 60/20/20 stratified train/validation/test split."""

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return (
        X_train,
        X_val,
        X_test,
        X_train_val,
        y_train,
        y_val,
        y_test,
        y_train_val,
    )
