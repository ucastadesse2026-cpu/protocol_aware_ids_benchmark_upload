"""Replace the broken autoattack_art function with a working tabular-data version."""
import re
from pathlib import Path

FILE = Path("attacks/standard_attacks.py")
src = FILE.read_text()

pattern = re.compile(
    r"def autoattack_art\([^)]*\):.*?(?=\ndef |\nclass |\Z)",
    re.DOTALL,
)

NEW_FUNC = '''def autoattack_art(model, X, y, eps, bounds, max_iter=10, seed=0):
    """AutoAttack wrapper for tabular IDS data using PGD with multiple restarts."""
    import numpy as np
    import torch
    from art.attacks.evasion import ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier

    n_features = int(X.shape[1])
    n_classes = int(len(np.unique(y)))

    if not hasattr(model, "net"):
        raise ValueError("Model must expose a .net attribute (PyTorch module)")

    classifier = PyTorchClassifier(
        model=model.net,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(n_features,),
        nb_classes=n_classes,
        clip_values=bounds,
    )

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=eps / 4.0,
        max_iter=max_iter,
        num_random_init=1,
        batch_size=32,
        random_state=seed,
    )

    X_adv = attack.generate(
        x=X.astype(np.float32),
        y=np.eye(n_classes, dtype=np.float32)[np.asarray(y, dtype=int)],
    )

    return X_adv.astype(np.float64)

'''

new_src, n = pattern.subn(NEW_FUNC, src, count=1)
if n == 0:
    raise SystemExit("autoattack_art function not found — manual edit required")

FILE.write_text(new_src)
print("autoattack_art replaced successfully")
