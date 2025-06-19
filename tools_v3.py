import numpy as np


def nto1(value, grid):
    """
    数値から最も近いグリッドインデックスを返す

    Parameters:
    -----------
    value : float or array
        変換したい値
    grid : ndarray
        グリッド配列

    Returns:
    --------
    int or ndarray
        最も近いグリッドインデックス 値がグリッドの範囲内に収まるように調整
    """
    # 値がグリッドの範囲内に収まるように調整
    aux = np.minimum(value, grid[-1])
    aux = np.maximum(aux, grid[0])

    # グリッドステップの計算
    step = (grid[-1] - grid[0]) / (len(grid) - 1)

    # インデックスの計算
    ind = np.round(((aux - grid[0]) / step)).astype(int)

    return ind


def fn_utility(value, gamma):
    """効用関数"""
    if gamma == 1:
        # gamma=1の時はlog関数を使用
        return np.log(np.maximum(value, 1e-10))  # 小さな正の値で置き換え
    else:
        value_safe = np.maximum(value, 1e-10)  # 小さな正の値で置き換え
        return (value_safe**(1-gamma))/(1-gamma)


def environment_retire(util, wage_retire, invest, portret, w_quad, func_value, discount, idx_valid_cons):
    """
    退職後における価値関数の効率的な実装

    Parameters:
    -----------
    util : np.ndarray
        今期の消費効用 (グリッド)
    wage_retire : float
        退職時の賃金
    invest : np.ndarray
        今期投資額(X_t - C_t)
    portret : np.ndarray
        ポートリターン
    w_quad : np.ndarray
        ガウス・エルミート求積法のノードウェイト
    func_value : callable
        翌期の価値関数 (の近似)
    discount : float
        割引率
    idx_valid_cons : np.ndarray
        適切な消費水準の消費グリッド番号

    Returns:
    --------
    tuple
        (最大価値, 最適消費インデックス, 最適投資比率インデックス)
    """
    # 有効な消費のみを計算して効率化
    if len(idx_valid_cons) == 0:
        raise ValueError("No valid consumption levels found")

    invest_valid = invest[idx_valid_cons]

    # ブロードキャストを使用して計算を効率化
    asset_grow = invest_valid[:, np.newaxis,
                              np.newaxis] * portret[np.newaxis, :, :]

    # 次期のキャッシュ計算
    cash_next = wage_retire + asset_grow

    # 初期化：すべての要素を-無限大に設定
    value_today = np.full((len(idx_valid_cons), portret.shape[0]), -np.inf)

    # 次期の価値関数評価
    value_next = func_value(cash_next)

    # ガウス・エルミート求積法での期待値計算
    exp_value = np.sum(value_next * w_quad, axis=2)

    # 今期の価値関数計算
    value_today = util[idx_valid_cons, np.newaxis] + discount * exp_value

    # 最適値と政策を見つける
    idx_flat = np.argmax(value_today)
    idx_best_cons_rel, idx_best_alpha = np.unravel_index(
        idx_flat, value_today.shape)

    # 相対インデックスから絶対インデックスへ変換
    idx_best_cons = idx_valid_cons[idx_best_cons_rel]

    value_today_max = value_today[idx_best_cons_rel, idx_best_alpha]

    return value_today_max, idx_best_cons, idx_best_alpha


def environment_work(util, wage_l_t, grid_wage_p, invest, portret, w_quad, func_value, discount, idx_valid_cons):
    """
    退職前における価値関数の効率的な実装

    Parameters:
    -----------
    util : np.ndarray
        今期の消費効用 (グリッド)
    wage_l_t : np.ndarray
        t歳のときの一時的ショック込(永続的ショックは抜き)の所得. shape=(n_quad,)
    grid_wage_p : np.ndarray
        永続的ショックの成長率グリッド. shape=(n_quad,n_quad)だが、
        1次元目は賃金の永続的ショックそのものの確率成分を表し、
        2次元目は賃金と株価の相関を表すための株式の確率成分を表す。
    invest : np.ndarray
        今期投資額(X_t - C_t)
    portret : np.ndarray
        ポートリターン
    w_quad : np.ndarray
        ガウス・エルミート求積法のノードウェイト
    func_value : callable
        翌期の価値関数 (の近似)
    discount : float
        割引率
    idx_valid_cons : np.ndarray
        適切な消費水準の消費グリッド番号

    Returns:
    --------
    tuple
        (最大価値, 最適消費インデックス, 最適投資比率インデックス)
    """
    # 有効な消費のみを計算して効率化
    if len(idx_valid_cons) == 0:
        raise ValueError("No valid consumption levels found")

    invest_valid = invest[idx_valid_cons]

    # ブロードキャストを使用して計算を効率化
    asset_grow = invest_valid[:, np.newaxis,
                              np.newaxis] * portret[np.newaxis, :, :]

    # 次期のキャッシュ計算
    cash_next = np.full(list(asset_grow.shape[:2]) + [len(w_quad)] * 3, np.nan)
    for idx_w_t in range(len(w_quad)):
        for idx_w_p in range(len(w_quad)):
            for idx_stock in range(len(w_quad)):
                cash_next[:, :, idx_w_t, idx_w_p, idx_stock] = (asset_grow[:, :, idx_stock]
                                                                + wage_l_t[idx_w_t] * np.exp(grid_wage_p[idx_w_p, idx_stock]))

    # 次期の価値関数評価
    value_next = func_value(cash_next)

    # ガウス・エルミート求積法での期待値計算 (TODO: 美しくない)
    exp_value = np.full(asset_grow.shape[:2], 0.0)
    for idx_w_t in range(len(w_quad)):
        for idx_w_p in range(len(w_quad)):
            for idx_stock in range(len(w_quad)):
                exp_value[:, :] = (exp_value[:, :]
                                   + w_quad[idx_w_t] * w_quad[idx_w_p] * w_quad[idx_stock] * value_next[:, :, idx_w_t, idx_w_p, idx_stock])

    # 今期の価値関数計算
    value_today = util[idx_valid_cons, np.newaxis] + discount * exp_value

    # 最適値と政策を見つける
    idx_flat = np.argmax(value_today)
    idx_best_cons_rel, idx_best_alpha = np.unravel_index(
        idx_flat, value_today.shape)

    # 相対インデックスから絶対インデックスへ変換
    idx_best_cons = idx_valid_cons[idx_best_cons_rel]

    value_today_max = value_today[idx_best_cons_rel, idx_best_alpha]

    return value_today_max, idx_best_cons, idx_best_alpha


def make_grid_maliar(N, theta, w_min, w_max):
    """
    Maliar et al. (2010)のグリッドの切り方
    """
    return np.array([w_min + (((i - 1) / (N - 1))**theta) * (w_max - w_min) for i in range(1, N+1)])
