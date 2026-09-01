"""
cluswisard.py -- Redes neurais sem peso WiSARD e ClusWiSARD.

WiSARD (Aleksander, Thomas & Bowden, 1984) e um classificador n-tupla:
a entrada binaria (retina) e particionada aleatoriamente em N tuplas de n
bits; cada tupla enderaca uma RAM de 2^n posicoes. Treinar e escrever 1 no
endereco visitado; classificar e contar quantas RAMs respondem, produzindo a
*resposta* do discriminador. Nao ha pesos nem gradiente -- o aprendizado e
uma unica escrita em memoria por tupla e por exemplo.

ClusWiSARD (Cardoso et al., 2016) generaliza o modelo permitindo que uma
mesma classe seja representada por *varios* discriminadores. Ao treinar, o
exemplo e enviado ao discriminador da classe que melhor responde a ele; se
nenhum atinge um limiar minimo de resposta, cria-se um novo discriminador.
Cada discriminador passa a representar um agrupamento (cluster) interno da
classe -- util quando uma classe e multimodal, como "eco de embarcacao" em
distancias e tamanhos muito diferentes.

O mesmo mecanismo, sem rotulos, transforma a rede em um *clusterizador*:
`ClusWiSARDClusterer` cria discriminadores sob demanda e devolve, para cada
exemplo, o indice do discriminador que o absorveu.

Implementacao vetorizada com RAMs densas (matriz N x 2^n de contadores), o
que permite classificar milhoes de amostras por lote.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Codificacao
# ---------------------------------------------------------------------------


def thermometer(v, lo, hi, levels):
    """Codificacao termometro: v -> `levels` bits monotonicos.

    Preserva a ordem da grandeza (bits vizinhos representam valores
    proximos), propriedade essencial para que a resposta da WiSARD decaia
    suavemente com a diferenca numerica entre exemplos.
    """
    v = np.asarray(v, np.float64)
    t = np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    k = np.arange(levels, dtype=np.float64) / levels
    return (t[..., None] > k).astype(np.uint8)


def multi_threshold(v, thresholds):
    """Binarizacao por um vetor de limiares (termometro nao uniforme)."""
    v = np.asarray(v)
    return (v[..., None] > np.asarray(thresholds)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Discriminador
# ---------------------------------------------------------------------------


_MIX = np.uint64(0x9E3779B97F4A7C15)
_ODD = np.uint64(0xBF58476D1CE4E5B9)


class Mapping:
    """Particao pseudoaleatoria da retina em N tuplas de n bits.

    Quando `2**tuple_size` excede `2**table_bits`, o endereco de n bits e
    reduzido por uma funcao de espalhamento multiplicativa (hashing) ao
    tamanho fisico da RAM. Isso permite tuplas grandes -- necessarias para que
    a WiSARD capture padroes espaciais extensos da retina -- com memoria
    constante, ao custo de colisoes raras e uniformemente distribuidas.
    """

    def __init__(self, n_bits: int, tuple_size: int, seed: int = 0, table_bits: int = 18,
                 n_ram: int | None = None):
        self.n_bits = int(n_bits)
        self.tuple_size = int(tuple_size)
        self.table_bits = int(min(table_bits, self.tuple_size))
        self.hashed = self.tuple_size > self.table_bits
        rng = np.random.default_rng(seed)
        if n_ram is None:
            n_ram = max(1, self.n_bits // self.tuple_size)
        need = int(n_ram) * self.tuple_size
        idx = np.concatenate([rng.permutation(self.n_bits)
                              for _ in range(int(np.ceil(need / self.n_bits)))])[:need]
        self.map = idx.reshape(int(n_ram), self.tuple_size)
        self.n_ram = int(n_ram)
        self.size = 1 << self.table_bits
        self._flat = self.map.ravel()
        self.pow2 = (1 << np.arange(self.tuple_size)).astype(np.int64)
        self._salt = (rng.integers(1, 2 ** 62, self.n_ram) | 1).astype(np.uint64)
        self._shift = np.uint64(64 - self.table_bits)

    def address(self, B: np.ndarray) -> np.ndarray:
        """B (n_amostras, n_bits) uint8 -> enderecos (n_amostras, n_ram) int64."""
        G = B[:, self._flat].reshape(B.shape[0], self.n_ram, self.tuple_size)
        P8 = np.packbits(G, axis=-1, bitorder="little").astype(np.int64)
        a = P8[:, :, 0].copy()
        for k in range(1, P8.shape[2]):
            a |= P8[:, :, k] << (8 * k)
        if not self.hashed:
            return a
        u = (a.astype(np.uint64) + self._salt[None, :]) * _MIX
        u ^= u >> np.uint64(29)
        u *= _ODD
        return (u >> self._shift).astype(np.int64)


class Discriminator:
    """Conjunto de N RAMs densas de 2^n posicoes (contadores de acesso)."""

    __slots__ = ("mp", "ram", "n_trained", "n_written")

    def __init__(self, mp: Mapping, dtype=np.uint16):
        self.mp = mp
        self.ram = np.zeros((mp.n_ram, mp.size), dtype)
        self.n_trained = 0
        self.n_written = 0

    @property
    def occupancy(self) -> float:
        """Fracao de posicoes de RAM ja escritas -- probabilidade a priori de
        uma resposta positiva por acaso."""
        return self.n_written / (self.mp.n_ram * self.mp.size)

    def train(self, addr: np.ndarray):
        """addr (B, n_ram) -- escreve (incrementa) os enderecos visitados."""
        rows = np.repeat(np.arange(self.mp.n_ram)[None, :], len(addr), 0).ravel()
        np.add.at(self.ram, (rows, addr.ravel()), 1)
        self.n_trained += len(addr)
        self.n_written = int(np.count_nonzero(self.ram))

    def train_one(self, addr_row: np.ndarray):
        idx = (np.arange(self.mp.n_ram), addr_row)
        r = self.ram[idx]
        self.n_written += int((r == 0).sum())
        self.ram[idx] = np.minimum(r.astype(np.int32) + 1,
                                   np.iinfo(self.ram.dtype).max).astype(self.ram.dtype)
        self.n_trained += 1

    def response(self, addr: np.ndarray, bleach: int = 1) -> np.ndarray:
        """Fracao de RAMs cujo conteudo no endereco visitado atinge `bleach`."""
        v = self.ram[np.arange(self.mp.n_ram)[None, :], addr]
        return (v >= bleach).sum(axis=1) / self.mp.n_ram

    def response_one(self, addr_row: np.ndarray, bleach: int = 1) -> float:
        v = self.ram[np.arange(self.mp.n_ram), addr_row]
        return float((v >= bleach).sum()) / self.mp.n_ram


# ---------------------------------------------------------------------------
# ClusWiSARD supervisionado
# ---------------------------------------------------------------------------


class ClusWiSARD:
    """Classificador ClusWiSARD: varios discriminadores por classe.

    Parametros
    ----------
    n_bits      tamanho da retina binaria
    tuple_size  n, numero de bits por RAM (2^n posicoes por RAM)
    min_score   limiar de resposta abaixo do qual um novo discriminador
                pode ser criado para a classe
    threshold   intervalo de crescimento: so se cria discriminador a cada
                `threshold` exemplos vistos da classe
    limit       numero maximo de discriminadores por classe
    bleach      limiar de bleaching usado na resposta
    """

    def __init__(self, n_bits, tuple_size=14, min_score=0.35, threshold=64,
                 limit=12, bleach=1, seed=0, table_bits=18, n_ram=None,
                 balance=True, dtype=np.uint16):
        self.mp = Mapping(n_bits, tuple_size, seed, table_bits, n_ram)
        self.dtype = dtype
        self.min_score = float(min_score)
        self.threshold = int(threshold)
        self.limit = int(limit)
        self.bleach = int(bleach)
        self.balance = bool(balance)
        self.discs: dict[int, list[Discriminator]] = {}
        self.count: dict[int, int] = {}

    # -- treino ------------------------------------------------------------
    def fit(self, B: np.ndarray, y: np.ndarray):
        addr = self.mp.address(B)
        for i in range(len(addr)):
            self._fit_one(addr[i], int(y[i]))
        return self

    def _fit_one(self, a, c):
        D = self.discs.setdefault(c, [])
        n = self.count.get(c, 0)
        if not D:
            D.append(Discriminator(self.mp, self.dtype))
            D[0].train_one(a)
        else:
            r = [d.response_one(a, self.bleach) for d in D]
            if self.balance:
                # ganho sobre o acaso: impede que o discriminador mais treinado
                # (e portanto mais saturado) absorva indefinidamente os exemplos
                r = [ri - d.occupancy for ri, d in zip(r, D)]
            j = int(np.argmax(r))
            if r[j] < self.min_score and len(D) < self.limit and n % self.threshold == 0:
                d = Discriminator(self.mp, self.dtype)
                d.train_one(a)
                D.append(d)
            else:
                D[j].train_one(a)
        self.count[c] = n + 1

    # -- inferencia --------------------------------------------------------
    def scores(self, B: np.ndarray, batch=200_000) -> dict[int, np.ndarray]:
        """Resposta maxima por classe (dicionario classe -> vetor)."""
        out = {c: np.zeros(len(B)) for c in self.discs}
        for s in range(0, len(B), batch):
            e = min(s + batch, len(B))
            a = self.mp.address(B[s:e])
            for c, D in self.discs.items():
                r = np.zeros(e - s)
                for d in D:
                    np.maximum(r, d.response(a, self.bleach), out=r)
                out[c][s:e] = r
        return out

    def margin(self, B, pos=1, neg=0, batch=200_000) -> np.ndarray:
        """Diferenca de resposta (classe positiva - negativa) em [-1, 1]."""
        s = self.scores(B, batch)
        return s.get(pos, np.zeros(len(B))) - s.get(neg, np.zeros(len(B)))

    def predict(self, B, batch=200_000) -> np.ndarray:
        s = self.scores(B, batch)
        cs = sorted(s)
        M = np.stack([s[c] for c in cs], 1)
        return np.array(cs)[M.argmax(1)]

    @property
    def n_disc(self) -> dict:
        return {c: len(d) for c, d in self.discs.items()}


# ---------------------------------------------------------------------------
# ClusWiSARD nao supervisionado (clusterizador)
# ---------------------------------------------------------------------------


class ClusWiSARDClusterer:
    """Agrupamento por criacao dinamica de discriminadores.

    Cada discriminador representa um agrupamento. Um exemplo e absorvido pelo
    discriminador de maior resposta se esta atinge `min_score`; caso
    contrario, inicia um novo agrupamento. Como a codificacao termometro faz
    a resposta decair com a distancia entre exemplos, `min_score` atua como
    um raio de vizinhanca implicito no espaco de atributos.

    Ha uma passada opcional de consolidacao (`n_passes > 1`) que reatribui os
    exemplos aos discriminadores ja formados, estabilizando o resultado
    quanto a ordem de apresentacao.
    """

    def __init__(self, n_bits, tuple_size=12, min_score=0.55, limit=64,
                 bleach=1, n_passes=2, seed=0, table_bits=18, n_ram=None):
        self.mp = Mapping(n_bits, tuple_size, seed, table_bits, n_ram)
        self.min_score = float(min_score)
        self.limit = int(limit)
        self.bleach = int(bleach)
        self.n_passes = int(n_passes)
        self.discs: list[Discriminator] = []

    def fit_predict(self, B: np.ndarray) -> np.ndarray:
        self.discs = []
        if len(B) == 0:
            return np.zeros(0, int)
        addr = self.mp.address(B)
        lab = np.full(len(B), -1, int)
        for i in range(len(addr)):
            a = addr[i]
            if self.discs:
                r = [d.response_one(a, self.bleach) for d in self.discs]
                j = int(np.argmax(r))
                if r[j] >= self.min_score:
                    self.discs[j].train_one(a)
                    lab[i] = j
                    continue
            if len(self.discs) >= self.limit:
                r = [d.response_one(a, self.bleach) for d in self.discs]
                j = int(np.argmax(r))
                self.discs[j].train_one(a)
                lab[i] = j
                continue
            d = Discriminator(self.mp)
            d.train_one(a)
            self.discs.append(d)
            lab[i] = len(self.discs) - 1
        for _ in range(self.n_passes - 1):
            for i in range(len(addr)):
                r = [d.response_one(addr[i], self.bleach) for d in self.discs]
                lab[i] = int(np.argmax(r))
        return lab


def interval_code(v, lo, hi, delta, half_width):
    """Codificacao por campo receptivo: uma barra de largura 2*half_width
    centrada no valor.

    Diferentemente do termometro (que codifica ordem), essa codificacao produz
    sobreposicao de bits **apenas** entre valores proximos: dois exemplos
    compartilham bits enquanto |v1 - v2| < 2*half_width e nao compartilham
    nenhum alem disso. E o codigo adequado para usar a resposta da WiSARD como
    criterio de vizinhanca em um clusterizador, pois torna explicita a escala
    de distancia do agrupamento.
    """
    v = np.asarray(v, np.float64)
    n = int(np.ceil((hi - lo) / delta)) + 1
    c = np.clip(np.round((v - lo) / delta), 0, n - 1).astype(np.int64)
    w = max(int(round(half_width / delta)), 0)
    g = np.arange(n)[None, :]
    return ((g >= (c[:, None] - w)) & (g <= (c[:, None] + w))).astype(np.uint8)
