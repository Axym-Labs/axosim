COMMENT
    Arbor-compatible implementation of the ModelDB mechanism
    NMDA_CA1_pyr_SC.mod from model 2019343:
    https://modeldb.science/getModelFile?embed=True&file=2019343%2FModel%2FMods_Tomko%2FNMDA_CA1_pyr_SC.mod&model=2019343

    We keep the published/default source parameters here:
      tau1 = 3.3 ms, tau2 = 102.38 ms, mg = 1 mM, pf = 0.03.

    The original source comments state:
      - voltage dependence is from Jahr & Stevens 1990
      - time constants were Q10-corrected for 34 Celsius
      - pf separates calcium current from nonspecific current
ENDCOMMENT

NEURON {
    POINT_PROCESS NMDA_CA1_pyr_SC
    USEION ca READ eca WRITE ica
    RANGE tau1, tau2, e, mg, pf
    NONSPECIFIC_CURRENT i
    RANGE g
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau1 = 3.3 (ms)
    tau2 = 102.38 (ms)
    e = 0 (mV)
    mg = 1 (mM)
    pf = 0.03 (1)
}

ASSIGNED {
    g (uS)
    factor
}

STATE {
    A (uS)
    B (uS)
}

INITIAL {
    LOCAL tp
    A = 0
    B = 0
    tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor = -exp(-tp/tau1) + exp(-tp/tau2)
    factor = 1/factor
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = (B - A)*mgblock(v)
    i = g*(v - e)*(1 - pf)
    ica = g*(v - eca)*pf
}

DERIVATIVE state {
    A' = -A/tau1
    B' = -B/tau2
}

FUNCTION mgblock(v(mV)) {
    mgblock = 1 / (1 + exp(0.062 * -v) * (mg / 3.57))
}

NET_RECEIVE(weight (uS)) {
    A = A + weight*factor
    B = B + weight*factor
}
