"""Bloc d'accumulateur conditionnel (Alcyon, variante A, phase 1 : v7 fige).

Le modele de base (v7) est garde tel quel. On ajoute :
  - `cond_input` : un second transformeur de features, etroit (BLOCK neurones par perspective,
    produit par paires -> BLOCK/2), alimente par les memes entrees que v7 ;
  - `layer_stacks.l1_extra` : la contribution de ce bloc a la premiere couche dense, dans un
    parametre SEPARE de la couche de v7, initialise a zero, sans biais.
Premiere couche dense = l1 de v7 (entrees v7) + l1_extra (sortie du bloc x masque), le masque
valant 1 quand la position remplit la condition. Au depart, le modele donne exactement les evals
de v7. Avec `freeze_base`, seuls `cond_input` et `l1_extra` sont entraines : les positions ou la
condition est fausse gardent exactement les evals de v7.

A l'export, l1_extra se fusionne avec l1 en une seule couche plus large.

Conditions (calculees depuis les indices HalfKAv2_hm de la perspective blanche) :
  - "typ" : les deux camps n'ont pas les memes pieces (dames, tours ou pieces legeres en
    nombre different). ~13 % des positions, ~66 % de l'ecart d'erreur avec SF v14.
"""
import copy
from types import SimpleNamespace

import torch
from torch import nn

from .model import NNUEModel
from .modules import ComposedFeatureTransformer, get_feature_cls
from .modules.layer_stacks import LayerStacks
from .modules.stacked_linear import StackedLinear

THREAT_INPUTS = 60720  # Full_Threats precede HalfKAv2_hm^ dans l'indice combine
HALFKA_PLANES = 768    # 12 types de pieces x 64 cases


def piece_counts(white_indices: torch.Tensor) -> torch.Tensor:
    """(batch, 6 types, 2 couleurs) depuis les indices de la perspective blanche."""
    h = white_indices.long() - THREAT_INPUTS
    valid = h >= 0
    p_idx = torch.where(valid, (h % HALFKA_PLANES) // 64, torch.zeros_like(h))
    onehot = torch.nn.functional.one_hot(p_idx, 12) * valid.unsqueeze(-1)
    counts = onehot.sum(dim=1)  # (batch, 12) ; p_idx = (type-1)*2 + (couleur != blanc)
    return counts.view(-1, 6, 2)


def cond_mask(white_indices: torch.Tensor, cond: str) -> torch.Tensor:
    c = piece_counts(white_indices)
    if cond == "typ":
        q = c[:, 4, 0] != c[:, 4, 1]
        r = c[:, 3, 0] != c[:, 3, 1]
        minor = (c[:, 1, 0] + c[:, 2, 0]) != (c[:, 1, 1] + c[:, 2, 1])
        return q | r | minor
    raise ValueError(f"condition inconnue : {cond}")


class CondLayerStacks(LayerStacks):
    """LayerStacks dont la premiere couche recoit en plus la sortie du bloc (l1_extra)."""

    def forward(self, x, xb, ls_indices, fake_quantize_acts=True, fake_quantize_weights=True):
        l1c_ = self.l1(x, ls_indices, fake_quantize_weights) + self.l1_extra(xb, ls_indices, fake_quantize_weights)

        l1x_out = l1c_[:, -2].view(-1, 1) - l1c_[:, -1].view(-1, 1)
        l1x_ = l1c_
        l1_sqr = torch.pow(l1x_, 2.0)
        if fake_quantize_acts:
            l1_sqr = self.quantization.fake_quantize_ls_act(l1_sqr)
        l1_sqr = l1_sqr * (self.quantization.sqr_crelu_correction_factor)
        if fake_quantize_acts:
            l1x_ = self.quantization.fake_quantize_ls_act(l1x_)
        l1x_ = torch.cat([l1_sqr, l1x_], dim=1)
        l1x_ = self.quantization.clip_ls_act(l1x_)

        l2c_ = self.l2(l1x_, ls_indices, fake_quantize_weights)
        l2x_ = l2c_
        l2_sqr = torch.pow(l2x_, 2.0)
        if fake_quantize_acts:
            l2_sqr = self.quantization.fake_quantize_ls_act(l2_sqr)
        l2_sqr = l2_sqr * (self.quantization.sqr_crelu_correction_factor)
        if fake_quantize_acts:
            l2x_ = self.quantization.fake_quantize_ls_act(l2x_)
        l2x_ = torch.cat([l2_sqr, l2x_], dim=1)
        l2x_ = self.quantization.clip_ls_act(l2x_)

        l3_input = torch.cat([l1x_, l2x_], dim=1)
        l3c_ = self.output(l3_input, ls_indices, fake_quantize_weights)
        if fake_quantize_acts:
            l1x_out = self.quantization.fake_quantize_skip_act(l1x_out)
        l3x_ = l3c_ + l1x_out
        if fake_quantize_acts:
            l3x_ = self.quantization.fake_quantize_output(l3x_)
        return l3x_


class CondNNUEModel(NNUEModel):
    """NNUEModel + bloc conditionnel. Construit par `make_cond` a partir d'un modele existant."""

    def forward(
        self,
        us,
        them,
        white_indices,
        black_indices,
        piece_count,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        psqt_indices, layer_stack_indices = self.calculate_buckets(piece_count)
        l0_, wpsqt, bpsqt = self.forward_ft(
            us, them, white_indices, black_indices, psqt_indices,
            fake_quantize_acts, fake_quantize_weights,
        )
        lb, _, _ = self.cond_input(
            us, them, white_indices, black_indices, torch.zeros_like(psqt_indices),
            fake_quantize_acts, fake_quantize_weights,
        )
        mb = cond_mask(white_indices, self.cond_name).unsqueeze(1)
        m = mb.to(lb.dtype)
        if getattr(self, "variant", "A") in ("B", "W"):
            # Variante B : piles de couches propres aux positions sous condition (copies de v7
            # entrainables), v7 inchange ailleurs.
            base = self.layer_stacks(l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights)
            cond = self.cond_stacks(l0_, lb * m, layer_stack_indices, fake_quantize_acts, fake_quantize_weights)
            out = torch.where(mb, cond, base)
        else:
            out = self.layer_stacks(l0_, lb * m, layer_stack_indices, fake_quantize_acts, fake_quantize_weights)
        return out + (wpsqt - bpsqt) * (us - 0.5)

    @torch.no_grad()
    def clip_weights(self, include_input):
        super().clip_weights(include_input)
        if include_input:
            self.cond_input.clip_weights(self.quantization)
        lim = self.quantization.max_hidden_weight[0]
        if getattr(self, "variant", "A") in ("B", "W"):
            self.cond_stacks.l1_extra.linear.weight.clamp_(-lim, lim)
            # memes bornes que les piles de v7 (voir NNUEModel.clip_weights)
            for group in self.quantization.generate_weight_clipping_config(SimpleNamespace(layer_stacks=self.cond_stacks)):
                for p in group["params"]:
                    lo, hi = group["min_weight"], group["max_weight"]
                    if "virtual_params" in group:
                        v = group["virtual_params"]
                        ev = v.repeat(p.shape[0] // v.shape[0], p.shape[1] // v.shape[1])
                        p.data.copy_(torch.max(torch.min(p.data, hi - ev), lo - ev))
                    else:
                        p.data.clamp_(lo, hi)
        else:
            self.layer_stacks.l1_extra.linear.weight.clamp_(-lim, lim)

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        super().zero_virtual_weights()
        self.cond_input.zero_virtual_weights()
        if getattr(self, "variant", "A") in ("B", "W"):
            self.cond_stacks.zero_virtual_weights()


@torch.no_grad()
def widen_stacks(old: LayerStacks, quantization, L2n: int, L3n: int) -> LayerStacks:
    """Copie de `old` avec une premiere couche de L2n neurones et une deuxieme de L3n, qui donne
    exactement les memes sorties : les poids de v7 sont recopies, les nouveaux neurones ont des
    poids d'entree aleatoires (init par defaut) et des poids de sortie nuls. Les deux derniers
    neurones de la premiere couche (connexion directe vers la sortie) restent les deux derniers."""
    L2, L3, cnt = old.L2, old.L3, old.count
    new = LayerStacks(cnt, SimpleNamespace(L1=old.L1, L2=L2n, L3=L3n), quantization)
    m1 = lambda o: o if o < L2 - 2 else L2n - (L2 - o)          # neurones de l1
    for s in range(cnt):
        for o in range(L2):
            new.l1.linear.weight[s * L2n + m1(o)] = old.l1.linear.weight[s * L2 + o]
            new.l1.linear.bias[s * L2n + m1(o)] = old.l1.linear.bias[s * L2 + o]
    new.l1.factorized_linear.weight.zero_(); new.l1.factorized_linear.bias.zero_()
    for o in range(L2):
        new.l1.factorized_linear.weight[m1(o)] = old.l1.factorized_linear.weight[o]
        new.l1.factorized_linear.bias[m1(o)] = old.l1.factorized_linear.bias[o]
    # l2 : entrees = [carre(L2), lineaire(L2)] -> [carre(L2n), lineaire(L2n)]
    c2 = [part * L2n + m1(o) for part in (0, 1) for o in range(L2)]
    w2 = new.l2.linear.weight.view(cnt, L3n, 2 * L2n)
    w2[:, :, :] = 0.0 * w2  # colonnes nouvelles nulles
    nw = torch.randn(cnt, L3n - L3, 2 * L2n) * (1.0 / (2 * L2n) ** 0.5)
    w2[:, L3:, :] = nw                                             # nouveaux neurones de l2 : entrees aleatoires
    ow2 = old.l2.linear.weight.view(cnt, L3, 2 * L2)
    w2[:, :L3, c2] = ow2
    b2 = new.l2.linear.bias.view(cnt, L3n)
    b2[:, :L3] = old.l2.linear.bias.view(cnt, L3)
    b2[:, L3:] = 0.0
    # sortie : entrees = [carre(L2n), lineaire(L2n), carre(L3n), lineaire(L3n)]
    c3 = c2 + [2 * L2n + part * L3n + j for part in (0, 1) for j in range(L3)]
    w3 = new.output.linear.weight.view(cnt, 1, 2 * L2n + 2 * L3n)
    w3.zero_()
    w3[:, :, c3] = old.output.linear.weight.view(cnt, 1, 2 * L2 + 2 * L3)
    new.output.linear.bias.copy_(old.output.linear.bias)
    return new


@torch.no_grad()
def make_cond(model: NNUEModel, feature_name: str, block: int = 128, cond: str = "typ", freeze_base: bool = True,
              variant: str = "A") -> NNUEModel:
    """Transforme un NNUEModel en CondNNUEModel (en place) en gardant ses poids."""
    feature_cls = get_feature_cls(feature_name)
    model.cond_input = ComposedFeatureTransformer(feature_cls, block, 1, model.quantization)
    model.cond_input.init_weights()
    # Initialisation par copie de block/2 paires de neurones de v7 (le produit par paires associe
    # i et i + L1/2). Une init aleatoire donne des activations si faibles qu'elles tombent a 0 a la
    # quantification : bloc mort, aucun gradient (constate le 26/11).
    L1 = model.input.l1_size
    half = block // 2
    g = torch.Generator().manual_seed(1)
    sel = torch.randperm(L1 // 2, generator=g)[:half]
    src = torch.cat([sel, sel + L1 // 2])
    dst = torch.arange(block)
    for fc, fm in zip(model.cond_input.features, model.input.features):
        fc.weight[:, dst] = fm.weight[:, src]
        fc.weight[:, block:] = 0.0
        if hasattr(fc, "virtual_weight"):
            fc.virtual_weight[:, dst] = fm.virtual_weight[:, src]
            fc.virtual_weight[:, block:] = 0.0
    model.cond_input.bias[dst] = model.input.bias[src]
    model.cond_input.bias[block:] = 0.0
    model.cond_name = cond
    model.cond_block = block

    model.variant = variant
    if variant in ("B", "W"):
        model.cond_stacks = copy.deepcopy(model.layer_stacks)
        if variant == "W":
            model.cond_stacks = widen_stacks(model.cond_stacks, model.quantization, 128, 64)
        ls = model.cond_stacks
        prefix = "cond_stacks."
    else:
        ls = model.layer_stacks
        prefix = "layer_stacks."
    extra = StackedLinear(block, ls.l1.out_features, ls.count, model.quantization, "ls_l1")
    extra.linear.weight.zero_()
    extra.linear.bias.zero_()
    extra.linear.bias.requires_grad_(False)
    ls.l1_extra = extra
    ls.__class__ = CondLayerStacks

    model.freeze_base = freeze_base
    if freeze_base:
        for name, p in model.named_parameters():
            if variant in ("B", "W"):
                train = name.startswith("cond_input.") or (name.startswith("cond_stacks.") and name != "cond_stacks.l1_extra.linear.bias")
            else:
                train = name.startswith("cond_input.") or name == "layer_stacks.l1_extra.linear.weight"
            p.requires_grad_(train)

    model.__class__ = CondNNUEModel
    return model
