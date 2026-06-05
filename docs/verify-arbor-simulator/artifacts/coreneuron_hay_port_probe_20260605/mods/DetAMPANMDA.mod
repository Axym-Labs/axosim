TITLE Deterministic AMPA and NMDA receptor for CoreNEURON probe

COMMENT
Deterministic subset of ProbAMPANMDA2 used by the current Hay source-teacher
generator when Use=1, Dep=0, Fac=0, u0=0. Each event always increments the
AMPA and NMDA dual-exponential conductance states by the NetCon weight.
ENDCOMMENT

NEURON {
    POINT_PROCESS DetAMPANMDA
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA
    RANGE i, i_AMPA, i_NMDA, g_AMPA, g_NMDA, e, gmax, mg
    NONSPECIFIC_CURRENT i_AMPA, i_NMDA
}

PARAMETER {
    tau_r_AMPA = 0.3 (ms)
    tau_d_AMPA = 3.0 (ms)
    tau_r_NMDA = 2.0 (ms)
    tau_d_NMDA = 70.0 (ms)
    e = 0 (mV)
    mg = 1 (mM)
    gmax = 0.0004 (uS)
}

ASSIGNED {
    v (mV)
    i (nA)
    i_AMPA (nA)
    i_NMDA (nA)
    g_AMPA (uS)
    g_NMDA (uS)
    factor_AMPA
    factor_NMDA
    mggate
}

STATE {
    A_AMPA
    B_AMPA
    A_NMDA
    B_NMDA
}

INITIAL {
    LOCAL tp_AMPA, tp_NMDA

    A_AMPA = 0
    B_AMPA = 0
    A_NMDA = 0
    B_NMDA = 0

    tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA)
    tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA)

    factor_AMPA = 1/(-exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA))
    factor_NMDA = 1/(-exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA))
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    mggate = 1 / (1 + exp(0.08 (/mV) * -v) * (mg / 3.57 (mM)))
    g_AMPA = gmax*(B_AMPA-A_AMPA)
    g_NMDA = gmax*(B_NMDA-A_NMDA) * mggate
    i_AMPA = g_AMPA*(v-e)
    i_NMDA = g_NMDA*(v-e)
    i = i_AMPA + i_NMDA
}

DERIVATIVE state {
    A_AMPA' = -A_AMPA/tau_r_AMPA
    B_AMPA' = -B_AMPA/tau_d_AMPA
    A_NMDA' = -A_NMDA/tau_r_NMDA
    B_NMDA' = -B_NMDA/tau_d_NMDA
}

NET_RECEIVE (weight) {
    A_AMPA = A_AMPA + weight*factor_AMPA
    B_AMPA = B_AMPA + weight*factor_AMPA
    A_NMDA = A_NMDA + weight*factor_NMDA
    B_NMDA = B_NMDA + weight*factor_NMDA
}
