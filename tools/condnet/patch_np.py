"""Applique les modifications Alcyon a nnue-pytorch 4289208 (a lancer depuis la racine de np).

1. model/cond_model.py : bloc conditionnel.
2. lightning_module.configure_optimizers : les parametres figes (requires_grad False) sont retires
   des groupes de l'optimiseur (sinon la decroissance des poids de ranger21 les ferait deriver), et
   les parametres entrainables hors des groupes standard (bloc) sont ajoutes.
"""
import shutil, sys
shutil.copy(sys.argv[1], "model/cond_model.py")
p = "model/lightning_module.py"; s = open(p).read()
anchor = "        return self.optimizer_wrapper.configure_optimizers(train_params)"
add = '''        # Alcyon : retirer les parametres figes, ajouter les parametres hors groupes (bloc).
        for g in train_params:
            g["params"] = [q for q in g["params"] if q.requires_grad]
        seen = {id(q) for g in train_params for q in g["params"]}
        extra = [q for q in self.model.parameters() if q.requires_grad and id(q) not in seen]
        train_params.append({"params": [q for q in extra if q.dim() > 1], "lr": LRs[0], "weight_decay": ft_wd})
        train_params.append({"params": [q for q in extra if q.dim() <= 1], "lr": LRs[1], "weight_decay": 0.0})
        train_params = [g for g in train_params if g["params"]]
'''
if "Alcyon : retirer les parametres figes" not in s:
    assert anchor in s
    s = s.replace(anchor, add + anchor)
    open(p, "w").write(s)
print("np modifie")
