# -*- coding: utf-8 -*-
import csv
import datetime as dt
import os

import numpy as np
# 相对路径导入
try:
    from . import thermoregulation as threg
    from . import matrix
    from .matrix import NUM_NODES, INDEX, VINDEX, BODY_NAMES, remove_bodyname
    from .comfmod import preferred_temp
    from . import construction as cons
    from .construction import _BSAst
    from .params import ALL_OUT_PARAMS, show_outparam_docs
# 绝对路径导入（调试用）
except ImportError:
    from jos3 import thermoregulation as threg
    from jos3 import matrix
    from jos3.matrix import NUM_NODES, INDEX, VINDEX, BODY_NAMES, remove_bodyname
    from jos3.comfmod import preferred_temp
    from jos3 import construction as cons
    from jos3.construction import _BSAst
    from jos3.params import ALL_OUT_PARAMS, show_outparam_docs

class JOS3():
    """
    JOS-3是一个人体体温调节的数值模拟模型。

    参数说明
    -------
    height : float, 可选
        身高 [米]。默认值为1.72。
    weight : float, 可选
        体重 [千克]。默认值为74.43。
    fat : float, 可选
        体脂率 [%]。默认值为15。
    age : int, 可选
        年龄 [岁]。默认值为20。
    sex : str, 可选
        性别，"male"（男性）或 "female"（女性）。默认值为"male"。
    ci : float, 可选
        心指数 [升/分钟/平方米]。默认值为2.6432。
    bmr_equation : str, 可选
        选择基础代谢率（BMR）计算公式。默认值为"harris-benedict"。
        若使用日本人的公式，输入"japanese"。
    bsa_equation : str, 可选
        选择体表面积（BSA）计算公式。
        可选值："dubois", "fujimoto", "kruazumi", "takahira"。
        默认值为"dubois"。
    ex_output : None, list 或 "all", 可选
        若需要额外的输出参数，以列表形式指定参数，如["BFsk", "BFcr", "Tar"]。
        若ex_output为"all"，则输出所有参数。
        默认值为None，仅输出主要参数（如全身皮肤温度）。


    设置器与获取器
    -------
    环境条件的输入参数通过设置器（Setter）形式设置。
    若为不同身体部位设置不同条件，需输入列表。
    列表输入必须为17个元素，对应以下身体部位：
    "Head", "Neck", "Chest", "Back", "Pelvis", "LShoulder", "LArm", "LHand",
    "RShoulder", "RArm", "RHand", "LThigh", "LLeg", "LFoot", "RThigh", "RLeg", "RFoot"

    Ta : float 或 list
        空气温度 [摄氏度]。
    Tr : float 或 list
        平均辐射温度 [摄氏度]。
    To : float 或 list
        操作温度 [摄氏度]。
    Va : float 或 list
        空气流速 [米/秒]。
    RH : float 或 list
        相对湿度 [%]。
    Icl : float 或 list
        服装热阻 [clo]。
    PAR : float
        身体活动系数 [-]。
        等于代谢率与基础代谢率的比值。
        休息时的PAR为1.2。
    posture : str
        选择姿势：standing（站立）、sitting（坐姿）或lying（躺姿）。
    bodytemp : numpy.ndarray (85,)
        JOS-3模型的所有体部温度


    获取器（Getter）
    -------
    JOS3提供一些实用的获取器以查看当前参数。

    BSA : numpy.ndarray (17,)
        各身体部位的体表面积 [平方米]。
    Rt : numpy.ndarray (17,)
        各身体部位皮肤与环境之间的干热阻 [K·平方米/W]。
    Ret : numpy.ndarray (17,)
        各身体部位皮肤与环境之间的湿热阻 [Pa·平方米/W]。
    Wet : numpy.ndarray (17,)
        各身体部位的皮肤湿润度 [-]。
    WetMean : float
        全身平均皮肤湿润度 [-]。
    TskMean : float
        全身平均皮肤温度 [°C]。
    Tsk : numpy.ndarray (17,)
        各身体部位的皮肤温度 [°C]。
    Tcr : numpy.ndarray (17,)
        各身体部位的核心温度 [°C]。
    Tcb : numpy.ndarray (1,)
        核心体温 [摄氏度]。
    Tar : numpy.ndarray (17,)
        各身体部位的动脉温度 [°C]。
    Tve : numpy.ndarray (17,)
        各身体部位的静脉温度 [°C]。
    Tsve : numpy.ndarray (12,)
        各身体部位的浅静脉温度 [°C]。
    Tms : numpy.ndarray (2,)
        头部和躯干的肌肉温度 [°C]。
    Tfat : numpy.ndarray (2,)
        头部和躯干的脂肪温度 [°C]。
    BMR : float
        基础代谢率 [W/平方米]。


    """


    def __init__(
            self,
            height=1.72,
            weight=74.43,
            fat=15,
            age=20,
            sex="male",
            ci=2.59,
            bmr_equation="harris-benedict",
            bsa_equation="dubois",
            ex_output=None,
            ):

        # 初始化模型参数
        self._height = height  # 身高
        self._weight = weight  # 体重
        self._fat = fat  # 体脂率
        self._sex = sex  # 性别
        self._age = age  # 年龄
        self._ci = ci  # 心指数
        self._bmr_equation = bmr_equation  # BMR计算公式
        self._bsa_equation = bsa_equation  # BSA计算公式（表面积）
        self._ex_output = ex_output  # 额外输出参数设置
        self.total_weight_loss = 0  # 初始化累计体重损失
        # 体表面积 [平方米]
        self._bsa_rate = cons.bsa_rate(height, weight, bsa_equation,)
        # 体表面积比例 [-]
        self._bsa = cons.localbsa(height, weight, bsa_equation,)
        # 血流量比例 [-]
        self._bfb_rate = cons.bfb_rate(height, weight, bsa_equation, age, ci)
        # 热导 [W/K]
        self._cdt = cons.conductance(height, weight, bsa_equation, fat,)
        # 热容 [J/K]
        self._cap = cons.capacity(height, weight, bsa_equation, age, ci)

        # 初始化核心温度设定值和皮肤温度设定值
        self.Tc_set_min = 36.5  # 核心温度设定值 (Havenith模型推荐值为37°C，但可以根据需要微调)
        self.Tsk_set_min = 34   # 皮肤温度设定值 (Havenith模型推荐值)

        # 设定温度 [°C]
        self.setpt_cr = np.ones(17)*37  # 核心
        self.setpt_sk = np.ones(17)*34  # 皮肤

        # 初始体温 [°C]
        self._bodytemp = np.ones(NUM_NODES) * 36

        # 输入条件的默认值
        self._ta = np.ones(17)*28.8  # 空气温度
        self._tr = np.ones(17)*28.8  # 平均辐射温度
        self._rh = np.ones(17)*50  # 相对湿度


        self._ret = None  # 服装蒸发热阻 [Pa·m2/W]
        self._va = np.ones(17)*0.1  # 空气流速
        self._clo = np.zeros(17)  # 服装热阻
        self._iclo = np.ones(17) * 0.45  # 服装蒸发热阻系数
        self._clo_includes_air_layer = False  # 输入的 clo 是否包含内部空气层
        self._ret_includes_air_layer = False  # 输入的 ret 是否包含内部空气层


        self._posture = "standing"  # 姿势
        self._hc = None  # 对流换热系数（手动设置）
        self._hr = None  # 辐射换热系数（手动设置）
        self.ex_q = np.zeros(NUM_NODES)  # 额外热量输入
        self._t = dt.timedelta(0)  # 累计时间
        self._cycle = 0  # 循环次数
        self.model_name = "JOS3"  # 模型名称
        # 模型选项设置
        self.options = {
                "nonshivering_thermogenesis": True,     # 是否考虑非颤抖产热
                "cold_acclimated": False,               # 是否冷适应
                "shivering_threshold": False,           # 颤抖阈值
                "limit_dshiv/dt": False,                # 是否限制颤抖产热变化率
                "bat_positive": False,                  # 是否考虑棕色脂肪
                "ava_zero": False,                      # 是否关闭动静脉吻合
                "shivering": False,}                    # 是否颤抖

        # 行军与负重条件的默认值
        self.load_mass = 0.0     # 负重 [kg]
        self.march_speed = 0.0   # 行进速度 [m/s]
        self.slope = 0.0         # 坡度 [%]
        self.snow_depth = 0.0    # 雪深 [cm]


        threg.PRE_SHIV = 0  # 重置颤抖相关参数
        self._history = []  # 存储模拟历史数据
        self._t = dt.timedelta(0)  # 累计时间（再次初始化）
        self._cycle = 0  # 循环次数（再次初始化）
        self._atmospheric_pressure = 101.33  # kPa. 用于计算hc和he

        # 重置设定温度
        dictout = self._reset_setpt()
        self._history.append(dictout)  # 保存初始模型参数

    def _reset_setpt(self):
        """
        通过迭代计算重置设定温度。
        注意：输入参数（Ta, Tr, RH, Va, Icl, PAR）和体温也会被重置。

        Returns
        -------
        JOS-3的参数 : dict
        """
        # 在PMV=0的环境下设定操作温度
        # PAR = 1.25
        # 1 met = 58.15 W/平方米
        met = self.BMR * 1.25 / 58.15  # 转换为[met]单位
        self.To = preferred_temp(met=met)  # 设定舒适温度
        self.RH = 50  # 相对湿度设为50%
        self.Va = 0.1  # 空气流速设为0.1m/s
        self.Icl = 0  # 服装热阻设为0
        self.Scl = 0  # 服装蒸发热阻设为0
        self.PAR = 1.25  # 身体活动系数设为1.25

        # 迭代计算
        self.options["ava_zero"] = True  # 关闭动静脉吻合
        for t in range(10):  # 迭代10次达到稳态
            dictout = self._run(dtime=60000, passive=True)  # 长时间被动暴露

        # 设定新的设定温度
        self.setpt_cr = self.Tcr                            # 以当前核心温度为设定值
        self.setpt_sk = self.Tsk                            # 以当前皮肤温度为设定值
        self.options["ava_zero"] = False                    # 恢复动静脉吻合功能



        return dictout


    def simulate(self, times, dtime=60, output=True):
        """
        执行JOS-3模型。

        参数
        ----------
        times : int
            模拟循环次数
        dtime : int 或 float, 可选
            时间步长 [秒]。默认值为60。
        output : bool, 可选
            若不记录参数，设为False。默认值为True。

        Returns
        -------
        None.

        """
        for t in range(times):                                      # 循环指定次数
            self._t += dt.timedelta(0, dtime)                   # 累计时间增加dtime秒
            self._cycle += 1                                        # 循环次数加1
            dictdata = self._run(dtime=dtime, output=output)        # 运行一次模型
            if output:                                              # 若需要记录
                self._history.append(dictdata)                      # 保存结果到历史记录



    def _run(self, dtime=60, passive=False, output=True):
        """
        运行一次模型并获取模型参数。

        参数
        ----------
        dtime : int 或 float, 可选
            时间步长 [秒]。默认值为60。
        passive : bool, 可选
            若运行被动模型（无体温调节），设为True。默认值为False。
        output : bool, 可选
            若不需要参数输出，设为False。默认值为True。
        Returns
        -------
        dictout : dictionary
            输出参数。

        """

        # 确保核心温度和皮肤温度已经正确初始化
        if not hasattr(self, 'Tcr') or not hasattr(self, 'Tsk'):
            raise ValueError("核心温度 Tcr 和皮肤温度 Tsk 未初始化")


        tcr = self.Tcr  # 获取当前核心温度
        tsk = self.Tsk  # 获取当前皮肤温度

        # 对流和辐射换热系数 [W/K·平方米]
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, tsk,), self._va)
        # 计算并平均相对对流换热系数
        hr = threg.fixed_hr(threg.rad_coef(self._posture,))
        # 计算并平均辐射换热系数
        # 手动设置（若已指定）
        if self._hc is not None:
            hc = self._hc                       # 使用手动设定的对流换热系数
        if self._hr is not None:
            hr = self._hr                       # 使用手动设定的辐射换热系数

        # 计算体重损失
        # 先初始化e_sweat和res_lh以避免后续错误
        e_sweat = 0
        res_lh = 0
        wlesk = (e_sweat + 0.06 * 0) / 2418  # 皮肤蒸发导致的体重损失（单位：kg）
        wleres = res_lh / 2418  # 呼吸蒸发导致的体重损失（单位：kg）

        # 计算体重损失（单位：克）
        total_weight_loss = (wlesk + wleres) * 1000  # 转换为克

        # 更新累计体重损失
        self.total_weight_loss += total_weight_loss  # 累加体重损失
        # 操作温度 [°C]，干热阻和湿热阻 [平方米·K/W]，[平方米·kPa/W]
        to = threg.operative_temp(self._ta, self._tr, hc, hr,)                                      # 计算操作温度
        r_t = threg.dry_r(
            hc,
            hr,
            self._clo,
            pt=self._atmospheric_pressure,
            clo_includes_air_layer=self._clo_includes_air_layer,
        )  # 计算干热阻
        # 计算湿热阻（单位：kPa·m2/W）
        r_et, r_ea, r_ecl, fcl = threg.wet_r(
            hc,
            self._clo,
            iclo=self._iclo,
            pt=self._atmospheric_pressure,
            ret_cl=self._ret,
            ret_cl_includes_air_layer=self._ret_includes_air_layer,
            return_components=True,
        )
        r_ea_eff = np.minimum(r_ea / fcl, r_et)

        #------------------------------------------------------------------
        # 体温调节
        #------------------------------------------------------------------
        # 体温调节设定点
        if passive:  # 被动模型（无调节）
            setpt_cr = tcr.copy()  # 核心温度设定点等于当前值
            setpt_sk = tsk.copy()  # 皮肤温度设定点等于当前值
        else:  # 自动调节模型
            setpt_cr = self.setpt_cr.copy()  # 使用预设核心设定点
            setpt_sk = self.setpt_sk.copy()  # 使用预设皮肤设定点

        # 设定点与实际体温的偏差
        err_cr = tcr - setpt_cr  # 核心温度偏差
        err_sk = tsk - setpt_sk  # 皮肤温度偏差
        # 皮肤湿润度 [-]，Esk, Emax, Esw [W]
        # 皮肤湿润度[-]，皮肤蒸发量、最大蒸发量、出汗量[W]
        # 进行出汗、散热、产热等计算

        wet, e_sk, e_max, e_sweat = threg.evaporation(
        err_cr, err_sk, self.Tsk,
        self._ta, self._rh, r_et,
        self._height, self._weight, self._bsa_equation, self._age)

        # 检查e_sweat是否为None，若是则赋值默认值
        if e_sweat is None:
            e_sweat = 0  # 默认值设为0，可根据需求修改

        # 计算核心温度误差 (Tc_error) 和皮肤温度误差 (Tsk_error)

        Tc_error = self.Tcr - self.Tc_set_min  # 核心温度误差
        Tsk_error = self.Tsk - self.Tsk_set_min  # 皮肤温度误差

        # 计算出汗量，基于Havenith模型的公式
        # 根据文献，出汗量的计算涉及温度差和湿度
        # 这里我们使用一个简单的线性模型，具体系数可以根据需要调整
        some_scaling_factor = 0.07  # 这个系数可以根据文献调整，代表汗腺反应程度
        e_sweat = (Tc_error + Tsk_error) * some_scaling_factor  # 调整出汗量的计算

        # 皮肤蒸发导致的体重损失
        wlesk = (e_sweat + 0.06 * e_max) / 2418
        # 呼吸蒸发导致的体重损失
        wleres = res_lh / 2418


        # 重新计算呼吸蒸发（这里需要确保res_lh已正确计算）
        # 先计算呼吸相关参数
        p_a = threg.antoine(self._ta) * self._rh / 100  # 计算水蒸气压
        # 假设qall已初步计算（实际应在产热计算后更新）
        qall = 0  # 临时初始值
        res_sh, res_lh = threg.resp_heatloss(self._ta[0], p_a[0], qall)  # 计算呼吸显热和潜热损失

        # 根据文献中的公式调整出汗相关计算
        wlesk = (e_sweat + 0.06 * e_max) / 2418  # 皮肤蒸发导致的体重损失
        wleres = res_lh / 2418  # 呼吸蒸发导致的体重损失（单位：kg）

        # 计算体重损失（单位：克）
        total_weight_loss = (wlesk + wleres) * 1000  # 转换为克



        # 皮肤血流量，基础皮肤血流量 [L/h]
        bf_sk = threg.skin_bloodflow(err_cr, err_sk,
                                     self._height, self._weight, self._bsa_equation, self._age, self._ci)

        # 手、足AVA血流量 [L/h]
        bf_ava_hand, bf_ava_foot = threg.ava_bloodflow(err_cr, err_sk,
            self._height, self._weight, self._bsa_equation, self._age, self._ci)
        if self.options["ava_zero"] and passive:
            bf_ava_hand = 0
            bf_ava_foot = 0

        # 颤抖产热 [W]
        mshiv = threg.shivering(
                err_cr, err_sk, tcr, tsk,
                self._height, self._weight, self._bsa_equation, self._age, self._sex, dtime,
                self.options,)

        # 非颤抖产热 [W]
        if self.options["nonshivering_thermogenesis"]:
            mnst = threg.nonshivering(err_cr, err_sk,
                self._height, self._weight, self._bsa_equation, self._age,
                self.options["cold_acclimated"], self.options["bat_positive"])
        else: # 不考虑非颤抖产热
            mnst = np.zeros(17)

        #------------------------------------------------------------------
        # 产热计算
        #------------------------------------------------------------------
        # 基础产热 [W]
        mbase = threg.local_mbase(
                self._height, self._weight, self._age, self._sex,
                self._bmr_equation,)
        mbase_all = sum([m.sum() for m in mbase])       # 总基础产热

        # 工作产热 [W]
        mwork = threg.local_mwork(mbase_all, self._par)

        # 核心、肌肉、脂肪、皮肤的总产热 [W]
        qcr, qms, qfat, qsk = threg.sum_m(mbase, mwork, mshiv, mnst,)
        qall = qcr.sum() + qms.sum() + qfat.sum() + qsk.sum()       # 总产热

        # 将修正的动态负重方程结果作为全身代谢率并等比例缩放各层代谢热
        target_met = self.calculate_metabolic_rate()
        if target_met is not None and target_met > 0 and qall > 0:
            scale = target_met / qall
            qcr *= scale
            qms *= scale
            qfat *= scale
            qsk *= scale
            qall = target_met

        #------------------------------------------------------------------
        # 其他计算
        #------------------------------------------------------------------
        # 核心、肌肉、脂肪的血流量 [L/h]
        bf_cr, bf_ms, bf_fat = threg.crmsfat_bloodflow(mwork, mshiv,
            self._height, self._weight, self._bsa_equation, self._age, self._ci)

        # 呼吸散热
        p_a = threg.antoine(self._ta)*self._rh/100                              # 计算水蒸气压
        res_sh, res_lh = threg.resp_heatloss(self._ta[0], p_a[0], qall)         # 计算呼吸显热和潜热损失

        # 显热损失 [W]
        shlsk = (tsk - to) / r_t * self._bsa                                    # 皮肤显热损失

        # 心输出量 [L/h]
        co = threg.sum_bf(
                bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot)          # 总血流量（心输出量）

        # 蒸发导致的体重损失率 [g/sec]
        wlesk = (e_sweat + 0.06*e_max) / 2418                                   # 皮肤蒸发导致的体重损失
        wleres = res_lh / 2418                                                  # 呼吸蒸发导致的体重损失

        #------------------------------------------------------------------
        # 矩阵计算
        #------------------------------------------------------------------
        # 矩阵A
        # (83, 83,) 数组     表示血流量
        bf_art, bf_vein = matrix.vessel_bloodflow(
                bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot
                )
        # 局部血流量矩阵
        bf_local = matrix.localarr(
                bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot
                )
        # 全身血流量矩阵
        bf_whole = matrix.wholebody(
                bf_art, bf_vein, bf_ava_hand, bf_ava_foot
                )
        # 血流量矩阵初始化
        arr_bf = np.zeros((NUM_NODES,NUM_NODES))
        arr_bf += bf_local                          # 加入局部血流量
        arr_bf += bf_whole                          # 加入全身血流量

        arr_bf /= self._cap.reshape((NUM_NODES,1)) # 单位从[W/K]转换为[/sec]
        arr_bf *= dtime # 单位从[/sec]转换为[-]

        arr_cdt = self._cdt.copy()
        arr_cdt /= self._cap.reshape((NUM_NODES,1)) # 单位从[W/K]转换为[/sec]
        arr_cdt *= dtime # 单位从[/sec]转换为[-]

        arrB = np.zeros(NUM_NODES)
        arrB[INDEX["skin"]] += 1/r_t*self._bsa
        arrB /= self._cap # 单位从[W/K]转换为[/sec]
        arrB *= dtime # 单位从[/sec]转换为[-]

        arrA_tria = -(arr_cdt + arr_bf)

        arrA_dia = arr_cdt + arr_bf
        arrA_dia = arrA_dia.sum(axis=1) + arrB
        arrA_dia = np.diag(arrA_dia)
        arrA_dia += np.eye(NUM_NODES)

        arrA = arrA_tria + arrA_dia
        arrA_inv = np.linalg.inv(arrA)

        # 矩阵Q [W] / [J/K] * [sec] = [-]
        # 产热项
        arrQ = np.zeros(NUM_NODES)
        arrQ[INDEX["core"]] += qcr                              # 核心产热
        arrQ[INDEX["muscle"]] += qms[VINDEX["muscle"]]          # 肌肉产热
        arrQ[INDEX["fat"]] += qfat[VINDEX["fat"]]               # 脂肪产热
        arrQ[INDEX["skin"]] += qsk                              # 皮肤产热

        # 呼吸项 [W]
        arrQ[INDEX["core"][2]] -= res_sh + res_lh # 胸腔核心减去呼吸总散热

        # 出汗项 [W]
        arrQ[INDEX["skin"]] -= e_sk                             # 皮肤节点减去皮肤蒸发散热

        # 额外热量输入 [W]
        arrQ += self.ex_q.copy()

        arrQ /= self._cap # 单位从[W]/[J/K]转换为[K/sec]
        arrQ *= dtime # 单位从[K/sec]转换为[K]

        # 边界温度矩阵 [°C]
        arr_to = np.zeros(NUM_NODES)
        arr_to[INDEX["skin"]] += to

        # 整体计算
        arr = self._bodytemp + arrB * arr_to + arrQ

        #------------------------------------------------------------------
        # 新的体温 [°C]
        #------------------------------------------------------------------
        self._bodytemp = np.dot(arrA_inv, arr)

        #------------------------------------------------------------------
        # 输出参数
        #------------------------------------------------------------------
        dictout = {}
        if output:  # 默认输出
            dictout["CycleTime"] = self._cycle  # 循环次数
            dictout["ModTime"] = self._t  # 模拟时间
            dictout["dt"] = dtime  # 时间步长
            dictout["TskMean"] = self.TskMean  # 平均皮肤温度
            dictout["Tsk"] = self.Tsk  # 各部位皮肤温度
            dictout["Tcr"] = self.Tcr  # 各部位核心温度
            dictout["WetMean"] = np.average(wet, weights=_BSAst)  # 平均皮肤湿润度
            dictout["Wet"] = wet  # 各部位皮肤湿润度
            dictout["Wle"] = (wlesk.sum() + wleres)  # 总水分损失
            dictout["CO"] = co  # 心输出量
            dictout["Met"] = qall  # 总代谢率
            dictout["RES"] = res_sh + res_lh  # 总呼吸散热
            dictout["THLsk"] = shlsk + e_sk  # 总皮肤散热


        detailout = {}
        if self._ex_output and output:  # 若需要额外输出
            detailout["Name"] = self.model_name  # 模型名称
            detailout["Height"] = self._height  # 身高
            detailout["Weight"] = self._weight  # 体重
            detailout["BSA"] = self._bsa  # 体表面积
            detailout["Fat"] = self._fat  # 体脂率
            detailout["Sex"] = self._sex  # 性别
            detailout["Age"] = self._age  # 年龄
            detailout["Setptcr"] = setpt_cr  # 核心设定温度
            detailout["Setptsk"] = setpt_sk  # 皮肤设定温度
            detailout["Tcb"] = self.Tcb  # 核心体温
            detailout["Tar"] = self.Tar  # 动脉温度
            detailout["Tve"] = self.Tve  # 静脉温度
            detailout["Tsve"] = self.Tsve  # 浅静脉温度
            detailout["Tms"] = self.Tms  # 肌肉温度
            detailout["Tfat"] = self.Tfat  # 脂肪温度
            detailout["To"] = to  # 操作温度
            detailout["Rt"] = r_t  # 干热阻
            detailout["Ret"] = (r_et * 1000).copy()  # 湿热阻 [Pa·m2/W]
            detailout["RetAirBoundary"] = (r_ea_eff * 1000).copy()  # 边界空气层湿阻 [Pa·m2/W]
            detailout["RetClothing"] = (r_ecl * 1000).copy()  # 服装自身湿阻 [Pa·m2/W]
            detailout["Fcl"] = fcl.copy()  # 服装面积系数
            detailout["Ta"] = self._ta.copy()  # 空气温度
            detailout["Tr"] = self._tr.copy()  # 辐射温度
            detailout["RH"] = self._rh.copy()  # 相对湿度
            detailout["Va"] = self._va.copy()  # 空气流速
            detailout["PAR"] = self._par  # 身体活动系数
            detailout["Icl"] = self._clo.copy()  # 服装热阻
            detailout["Esk"] = e_sk  # 皮肤蒸发量
            detailout["Emax"] = e_max  # 最大蒸发量
            detailout["Esweat"] = e_sweat  # 出汗量
            detailout["BFcr"] = bf_cr  # 核心血流量
            detailout["BFms"] = bf_ms[VINDEX["muscle"]]  # 肌肉血流量
            detailout["BFfat"] = bf_fat[VINDEX["fat"]]  # 脂肪血流量
            detailout["BFsk"] = bf_sk  # 皮肤血流量
            detailout["BFava_hand"] = bf_ava_hand  # 手AVA血流量
            detailout["BFava_foot"] = bf_ava_foot  # 足AVA血流量
            detailout["Mbasecr"] = mbase[0]  # 核心基础产热
            detailout["Mbasems"] = mbase[1][VINDEX["muscle"]]  # 肌肉基础产热
            detailout["Mbasefat"] = mbase[2][VINDEX["fat"]]  # 脂肪基础产热
            detailout["Mbasesk"] = mbase[3]  # 皮肤基础产热
            detailout["Mwork"] = mwork  # 工作产热
            detailout["Mshiv"] = mshiv  # 颤抖产热
            detailout["Mnst"] = mnst  # 非颤抖产热
            detailout["Qcr"] = qcr  # 核心总产热
            detailout["Qms"] = qms[VINDEX["muscle"]]  # 肌肉总产热
            detailout["Qfat"] = qfat[VINDEX["fat"]]  # 脂肪总产热
            detailout["Qsk"] = qsk  # 皮肤总产热
            detailout["WeightLoss"] = total_weight_loss  # 将体重损失添加到输出表格
            dictout["TotalWeightLoss"] = self.total_weight_loss  # 输出累计体重损失
            dictout["SHLsk"] = shlsk  # 皮肤显热损失
            dictout["LHLsk"] = e_sk  # 皮肤潜热损失
            dictout["RESsh"] = res_sh  # 呼吸显热损失
            dictout["RESlh"] = res_lh  # 呼吸潜热损失

        if self._ex_output == "all":    # 若输出所有参数
            dictout.update(detailout)   # 合并详细参数
        elif isinstance(self._ex_output, list):  # 若输出指定参数列表
            outkeys = detailout.keys()  # 详细参数的键
            for key in self._ex_output: # 遍历指定参数
                if key in outkeys:      # 若参数存在
                    dictout[key] = detailout[key]   # 添加到输出
        return dictout

    def dict_results(self):
        """
        以字典形式获取结果（可转换为pandas.DataFrame）
        Returns
        -------
        结果字典
        """
        if not self._history:       # 若历史记录为空
            print("模型无数据。")
            return None

        def check_word_contain(word, *args):
            """
            检查单词是否包含指定子串
            """
            boolfilter = False
            for arg in args:
                if arg in word:
                    boolfilter = True
            return boolfilter

        # 设置列标题
        # 若值为可迭代对象，添加身体部位名称作为后缀
        # 若值为非迭代的单一值，转换为可迭代对象
        key2keys = {}  # 键映射
        for key, value in self._history[0].items():
            try:
                length = len(value)  # 获取值的长度
                if isinstance(value, str):  # 若为字符串（虽可迭代但特殊处理）
                    keys = [key]  # 不添加后缀
                elif check_word_contain(key, "sve", "sfv", "superficialvein"):  # 浅静脉相关参数
                    keys = [key + BODY_NAMES[i] for i in VINDEX["sfvein"]]  # 添加对应身体部位后缀
                elif check_word_contain(key, "ms", "muscle"):  # 肌肉相关参数
                    keys = [key + BODY_NAMES[i] for i in VINDEX["muscle"]]  # 添加对应身体部位后缀
                elif check_word_contain(key, "fat"):  # 脂肪相关参数
                    keys = [key + BODY_NAMES[i] for i in VINDEX["fat"]]  # 添加对应身体部位后缀
                elif length == 17:  # 若为17个值（对应17个身体部位）
                    keys = [key + bn for bn in BODY_NAMES]  # 添加身体部位名称后缀
                else:  # 其他长度的可迭代对象
                    keys = [key + BODY_NAMES[i] for i in range(length)]  # 添加索引后缀
            except TypeError:  # 若值不可迭代（单一值）
                keys = [key]  # 转换为单元素列表
            key2keys.update({key: keys})  # 存储映射

        data = []
        for i, dictout in enumerate(self._history):  # 遍历历史记录
            row = {}
            for key, value in dictout.items():  # 遍历每条记录的键值对
                keys = key2keys[key]  # 获取映射的列名
                if len(keys) == 1:
                    values = [value]  # 单一值转换为列表
                else:
                    values = value  # 多值直接使用
                row.update(dict(zip(keys, values)))  # 添加到行数据
            data.append(row)  # 存储行数据
        # 转换为按列存储的字典
        outdict = dict(zip(data[0].keys(), [[] for i in range(len(data[0].keys()))]))
        for row in data:
            for k in data[0].keys():
                outdict[k].append(row[k])  # 按列收集数据
        return outdict

    # 在JOS3类中，确保代谢率计算方法正确接受模型实例的属性
    def calculate_metabolic_rate(self, dt=None):
        """
        计算代谢率 (W)，使用 Pandolf–Santee 方程并考虑雪地因素与动态响应。
        dt: 时间步长 (秒)。若为 None 或 0，则返回稳态代谢率；否则按一阶滞后动态响应计算代谢率。
        """
        # 获取模型参数
        W = self._weight  # 体重 (kg)
        L = self.load_mass  # 负载重量 (kg)
        v = self.march_speed  # 行走速度 (m/s)
        G = self.slope  # 坡度 (%)
        snow_depth = self.snow_depth  # 雪深 (cm)

        # 雪深转换为地面阻力系数 η（经验公式）:contentReference[oaicite:4]{index=4}
        eta = 1.3 + 0.08 * snow_depth  # 地面阻力系数 (无量纲)

        # Pandolf–Santee 方程三部分：静立、负载、行走:contentReference[oaicite:5]{index=5}
        M_stand = 1.5 * W  # 静立代谢 (W)
        M_load = 2.0 * (W + L) * (L / W) ** 2  # 负载代谢 (W)
        M_walk = eta * (W + L) * (1.5 * v ** 2 + 0.35 * v * G)  # 行走代谢 (W)
        P = M_stand + M_load + M_walk  # 总代谢率 (W)

        # 若未给定时间步长，则按稳态返回
        if not dt:
            self._prev_metabolic = P  # 保存当前稳态值
            return P

        # 第一阶动态响应：使用前一时刻代谢率进行指数平滑
        prev = getattr(self, '_prev_metabolic', None)
        if prev is None:
            # 初次调用时未初始化，则直接返回当前稳态
            self._prev_metabolic = P
            return P

        # 根据需求增加或减少选择不同时间常数 (τ_on/τ_off)
        tau = self.tau_on if P > prev else self.tau_off
        # 指数平滑计算新代谢率：M_new = prev + (P - prev)*(1 - exp(-dt/τ))
        M_new = prev + (P - prev) * (1 - np.exp(-dt / tau))
        self._prev_metabolic = M_new
        return M_new

    def to_csv(self, path=None, folder=None, unit=True, meaning=True):
        """
        将结果导出为csv格式。

        参数
        ----------
        path : str, 可选
            输出路径。若不使用默认文件名，可指定名称。
            默认值为None。
        folder : str, 可选
            输出文件夹。若使用含当前时间的默认文件名，
            仅需设置文件夹路径。
            默认值为None。
        unit : bool, 可选
            在csv文件中写入单位。默认值为True。
        meaning : bool, 可选
            在csv文件中写入参数含义。默认值为True。


        示例
        ----------
        >>> import jos3
        >>> model = jos3.JOS3()
        >>> model.simulate(60)
        >>> model.to_csv(folder="C:/Users/takahashi/desktop")
        """

        if path is None:
            nowtime = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = "{}_{}.csv".format(self.model_name, nowtime)
            if folder:
                os.makedirs(folder, exist_ok=True)
                path = folder + os.sep + path
        elif not ((path[-4:] == ".csv") or (path[-4:] == ".txt")):
            path += ".csv"
        dictout = self.dict_results()

        columns = [k for k in dictout.keys()]
        units = []
        meanings = []
        for col in columns:
            param, rbn = remove_bodyname(col)
            if param in ALL_OUT_PARAMS:
                u = ALL_OUT_PARAMS[param]["unit"]
                units.append(u)

                m = ALL_OUT_PARAMS[param]["meaning"]
                if rbn:
                    meanings.append(m.replace("身体部位", rbn))
                else:
                    meanings.append(m)
            else:
                units.append("")
                meanings.append("")

        with open(path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(columns))
            if unit: writer.writerow(units)
            if meaning: writer.writerow(meanings)
            for i in range(len(dictout["CycleTime"])):
                row = []
                for k in columns:
                    row.append(dictout[k][i])
                writer.writerow(row)


    #--------------------------------------------------------------------------
    # 设置器
    #--------------------------------------------------------------------------
    def _set_ex_q(self, tissue, value):
        """
        按组织名称设置额外热量输入。

        参数
        ----------
        tissue : str
            组织名称。"core", "skin", 或 "artery"等。若要设置头部肌肉和其他部位的核心，可设为"all_muscle"。
        value : int, float, array
            热量输入 [W]

        返回
        -------
        array
            模型的额外热量输入
        """
        self.ex_q[INDEX[tissue]] = value
        return self.ex_q


    #--------------------------------------------------------------------------
    # 设置器与获取器
    #--------------------------------------------------------------------------
    @property
    def Ta(self):
        """
        Getter

        Returns
        -------
        Ta : numpy.ndarray (17,)
            Air temperature [oC].
        """
        return self._ta
    @Ta.setter
    def Ta(self, inp):
        self._ta = _to17array(inp)

    @property
    def Iret(self):
        return None if self._ret is None else self._ret.copy()

    @Iret.setter
    def Iret(self, inp):
        if inp is None:
            self._ret = None
        else:
            array = _to17array(inp).astype(float)
            self._ret = array

    @property
    def IretIncludesAirLayer(self):
        """Return True when ``Iret`` values already include the air layer."""

        return self._ret_includes_air_layer

    @IretIncludesAirLayer.setter
    def IretIncludesAirLayer(self, value):
        self._ret_includes_air_layer = bool(value)
    @property
    def Tr(self):
        """
        Getter

        Returns
        -------
        Tr : numpy.ndarray (17,)
            Mean radiant temperature [oC].
        """
        return self._tr
    @Tr.setter
    def Tr(self, inp):
        self._tr = _to17array(inp)



    @property
    def To(self):
        """
        Getter

        Returns
        -------
        To : numpy.ndarray (17,)
            Operative temperature [oC].
        """
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, self.Tsk,), self._va)
        hr = threg.fixed_hr(threg.rad_coef(self._posture,))
        to = threg.operative_temp(self._ta, self._tr, hc, hr,)
        return to
    @To.setter
    def To(self, inp):
        self._ta = _to17array(inp)
        self._tr = _to17array(inp)

    @property
    def RH(self):
        """
        Getter

        Returns
        -------
        RH : numpy.ndarray (17,)
            Relative humidity [%].
        """
        return self._rh
    @RH.setter
    def RH(self, inp):
        self._rh = _to17array(inp)

    @property
    def Va(self):
        """
        Getter

        Returns
        -------
        Va : numpy.ndarray (17,)
            Air velocity [m/s].
        """
        return self._va
    @Va.setter
    def Va(self, inp):
        self._va = _to17array(inp)

    @property
    def posture(self):
        """
        Getter

        Returns
        -------
        posture : str
            Current JOS3 posture.
        """
        return self._posture
    @posture.setter
    def posture(self, inp):
        if inp == 0:
            self._posture = "standing"
        elif inp == 1:
            self._posture = "sitting"
        elif inp == 2:
            self._posture = "lying"
        elif type(inp) == str:
            if inp.lower() == "standing":
                self._posture = "standing"
            elif inp.lower() in ["sitting", "sedentary"]:
                self._posture = "sitting"
            elif inp.lower() in ["lying", "supine"]:
                self._posture = "lying"
        else:
            self._posture = "standing"
            print('posture must be 0="standing", 1="sitting" or 2="lying".')
            print('posture was set "standing".')

    @property
    def Icl(self):
        """
        Getter

        Returns
        -------
        Icl : numpy.ndarray (17,)
            Clothing insulation [clo].
        """
        return self._clo
    @Icl.setter
    def Icl(self, inp):
        self._clo = _to17array(inp)

    @property
    def IclIncludesAirLayer(self):
        """Return True when ``Icl`` values already include the air layer."""

        return self._clo_includes_air_layer

    @IclIncludesAirLayer.setter
    def IclIncludesAirLayer(self, value):
        self._clo_includes_air_layer = bool(value)


    @property
    def PAR(self):
        """
        Getter

        Returns
        -------
        PAR : float
            Physical activity ratio [-].
            This equals the ratio of metaboric rate to basal metablic rate.
            PAR of sitting quietly is 1.2.
        """
        # 计算代谢率P
        P = self.calculate_metabolic_rate()  # 通过模型实例调用计算方法
        # 返回代谢率与基础代谢率之比
        return P / self.BMR

    @PAR.setter
    def PAR(self, inp):
        self._par = inp


    @property
    def bodytemp(self):
        """
        Getter

        Returns
        -------
        bodytemp : numpy.ndarray (85,)
            All segment temperatures of JOS-3
        """
        return self._bodytemp
    @bodytemp.setter
    def bodytemp(self, inp):
        self._bodytemp = inp.copy()

    #--------------------------------------------------------------------------
    # Getter
    #--------------------------------------------------------------------------

    @property
    def BSA(self):
        """
        Getter

        Returns
        -------
        BSA : numpy.ndarray (17,)
            Body surface areas by local body segments [m2].
        """
        return self._bsa.copy()

    @property
    def Rt(self):
        """
        Getter

        Returns
        -------
        Rt : numpy.ndarray (17,)
            Dry heat resistances between the skin and ambience areas by local body segments [K.m2/W].
        """
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, self.Tsk,), self._va)
        hr = threg.fixed_hr(threg.rad_coef(self._posture,))
        return threg.dry_r(
            hc,
            hr,
            self._clo,
            clo_includes_air_layer=self._clo_includes_air_layer,
        )

    @property
    def Ret(self):
        """
        Getter

        Returns
        -------
        Ret : numpy.ndarray (17,)
            Wet (Evaporative) heat resistances between the skin and ambience areas by local body segments [Pa.m2/W].
        """
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, self.Tsk, ), self._va)
        return threg.wet_r(
            hc,
            self._clo,
            self._iclo,
            pt=self._atmospheric_pressure,
            ret_cl=self._ret,
            ret_cl_includes_air_layer=self._ret_includes_air_layer,
        ) * 1000

    @property
    def RetComponents(self):
        """Return a breakdown of evaporative resistances.

        Returns
        -------
        dict
            Dictionary with keys ``"total"``, ``"air_boundary"``, ``"clothing"``
            and ``"fcl"``. Resistances are expressed in ``Pa·m²/W`` while the
            clothing area factor is dimensionless.
        """

        hc = threg.fixed_hc(
            threg.conv_coef(self._posture, self._va, self._ta, self.Tsk, ), self._va
        )
        r_et, r_ea, r_ecl, fcl = threg.wet_r(
            hc,
            self._clo,
            iclo=self._iclo,
            pt=self._atmospheric_pressure,
            ret_cl=self._ret,
            ret_cl_includes_air_layer=self._ret_includes_air_layer,
            return_components=True,
        )
        r_ea_eff = np.minimum(r_ea / fcl, r_et)
        return {
            "total": (r_et * 1000).copy(),
            "air_boundary": (r_ea_eff * 1000).copy(),
            "clothing": (r_ecl * 1000).copy(),
            "fcl": fcl.copy(),
        }


    @property
    def Wet(self):
        """
        Getter

        Returns
        -------
        Wet : numpy.ndarray (17,)
            Skin wettedness on local body segments [-].
        """
        err_cr = self.Tcr - self.setpt_cr
        err_sk = self.Tsk - self.setpt_sk
        ret_kpa = self.Ret / 1000.0
        wet, *_ = threg.evaporation(err_cr, err_sk,
            self._ta, self._rh, ret_kpa, self._bsa_rate, self._age)
        return wet

    @property
    def WetMean(self):
        """
        Getter

        Returns
        -------
        WetMean : float
            Mean skin wettedness of the whole body [-].
        """
        wet = self.Wet
        return np.average(wet, weights=_BSAst)



    @property
    def TskMean(self):
        """
        Getter

        Returns
        -------
        TskMean : float
            Mean skin temperature of the whole body [oC].
        """
        return np.average(self._bodytemp[INDEX["skin"]], weights=_BSAst)

    @property
    def Tsk(self):
        """
        Getter

        Returns
        -------
        Tsk : numpy.ndarray (17,)
            Skin temperatures by the local body segments [oC].
        """
        return self._bodytemp[INDEX["skin"]].copy()

    @property
    def Tcr(self):
        """
        Getter

        Returns
        -------
        Tcr : numpy.ndarray (17,)
            Skin temperatures by the local body segments [oC].
        """
        return self._bodytemp[INDEX["core"]].copy()

    @property
    def Tcb(self):
        """
        Getter

        Returns
        -------
        Tcb : numpy.ndarray (1,)
            Core temperatures by the local body segments [oC].
        """
        return self._bodytemp[0].copy()

    @property
    def Tar(self):
        """
        Getter

        Returns
        -------
        Tar : numpy.ndarray (17,)
            Arterial temperatures by the local body segments [oC].
        """
        return self._bodytemp[INDEX["artery"]].copy()

    @property
    def Tve(self):
        """
        Getter

        Returns
        -------
        Tve : numpy.ndarray (17,)
            Vein temperatures by the local body segments [oC].
        """
        return self._bodytemp[INDEX["vein"]].copy()

    @property
    def Tsve(self):
        """
        Getter

        Returns
        -------
        Tsve : numpy.ndarray (12,)
            Superfical vein temperatures by the local body segments [oC].
        """
        return self._bodytemp[INDEX["sfvein"]].copy()

    @property
    def Tms(self):
        """
        Getter

        Returns
        -------
        Tms : numpy.ndarray (2,)
            Muscle temperatures of Head and Pelvis [oC].
        """
        return self._bodytemp[INDEX["muscle"]].copy()

    @property
    def Tfat(self):
        """
        Getter

        Returns
        -------
        Tfat : numpy.ndarray (2,)
            Fat temperatures of Head and Pelvis  [oC].
        """
        return self._bodytemp[INDEX["fat"]].copy()

    @property
    def bodyname(self):
        """
        Getter

        Returns
        -------
        bodyname : list
            JOS3 body names,
            "Head", "Neck", "Chest", "Back", "Pelvis",
            "LShoulder", "LArm", "LHand",
            "RShoulder", "RArm", "RHand",
            "LThigh", "LLeg", "LFoot",
            "RThigh", "RLeg" and "RFoot".
        """
        body = [
                "Head", "Neck", "Chest", "Back", "Pelvis",
                "LShoulder", "LArm", "LHand",
                "RShoulder", "RArm", "RHand",
                "LThigh", "LLeg", "LFoot",
                "RThigh", "RLeg", "RFoot",]
        return body

    @property
    def results(self):
        return self.dict_results()

    @property
    def BMR(self):
        """
        Getter

        Returns
        -------
        BMR : float
            Basal metabolic rate [W/m2].
        """
        bmr = threg.basal_met(
                self._height, self._weight, self._age,
                self._sex, self._bmr_equation,)
        return bmr / self.BSA.sum()


def _to17array(inp):
    """
    Make ndarray (17,).

    Parameters
    ----------
    inp : int, float, ndarray, list
        Number you make as 17array.

    Returns
    -------
    ndarray
    """
    try:
        if len(inp) == 17:
            array = np.array(inp)
        else:
            first_item = inp[0]
            array = np.ones(17)*first_item
    except:
        array = np.ones(17)*inp
    return array.copy()

if __name__ == "__main__":
    import jos3
