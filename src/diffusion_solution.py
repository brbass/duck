import numpy as np
from matplotlib import pyplot as plt

def diffusion_solution(x,  # Position
                       xb, # Boundary positions
                       xa, # Center position
                       df, # Diffusion coefficients
                       sa, # Absorption cross section
                       q,  # Isotropic internal source
                       j): # Isotropic boundary source
    # Get optimization symbols
    o5=sa[0];
    o3=df[0];
    o4=1./np.sqrt(o3);
    o6=np.sqrt(o5);
    o18=np.sqrt(o3);
    o10=df[1];
    o12=sa[1];
    o13=np.sqrt(o12);
    o19=2.*o18*o6;
    o21=o18*o6;
    o22=np.sqrt(o10);
    o11=1./np.sqrt(o10);
    o26=xb[0];
    o27=2.*o26*o4*o6;
    o28=np.exp(o27);
    o29=-1.+o19;
    o23=-(o13*o22);
    o24=o21+o23;
    o16=2.*o4*o6*xa;
    o17=np.exp(o16);
    o20=1.+o19;
    o30=o13*o22;
    o31=o21+o30;
    o34=2.*o13*o22;
    o14=2.*o11*o13*xa;
    o35=-1.+o34;
    o53=q[0];
    o58=-o53;
    o59=j[0];
    o60=4.*o5*o59;
    o61=o58+o60;
    o69=2.*x;
    o70=o26+o69;
    o71=o4*o6*o70;
    o37=xb[1];
    o38=2.*o11*o13*o37;
    o43=1.+o34;
    o63=2.*o26;
    o64=o63+x;
    o65=o4*o6*o64;
    o48=2.*xa;
    o49=o48+x;
    o50=o4*o49*o6;
    o91=o4*o6*xa;
    o92=o11*o13*xa;
    o93=o11*o13*o37;
    o96=q[1];
    o97=j[1];
    o98=-4.*o12*o97;
    o99=o96+o98;
    o107=o5*o96;
    o108=-(o12*o53);
    o109=o107+o108;
    o111=o69+xa;
    o112=o111*o4*o6;
    o15=np.exp(o14);
    o39=np.exp(o38);
    o140=o11*o13*o49;
    o147=2.*o37;
    o148=o147+x;
    o149=o11*o13*o148;
    o55=o26*o4*o6;
    o159=-4.*o5*o59;
    o160=o159+o53;
    o170=o37+o69;
    o171=o11*o13*o170;
    o177=-o96;
    o178=4.*o12*o97;
    o179=o177+o178;
    o184=o11*o111*o13;
    o193=-(o5*o96);
    o194=o12*o53;
    o195=o193+o194;

    # Get solution
    if x < xb[0]:
        return 0.
    elif x < xa:
        return (1.*np.exp(-(o4*o6*x))*(-(o109*o20*o22*o35*np.exp(o112+o14))+o109*o20*o22*o43*np.exp(o112+o38)+o13*o20*o24*o35*o53*np.exp(o14+o50)+o13*o20*o31*o43*o53*np.exp(o38+o50)+o13*o24*o35*o61*np.exp(o14+o16+o55)+o13*o31*o43*o61*np.exp(o38+o4*(o26+o48)*o6)-o13*o29*o31*o35*o53*np.exp(o14+o65)+o13*o29*o43*o53*(o30-o18*o6)*np.exp(o38+o65)+o13*o31*o35*o61*np.exp(o14+o71)+o13*o24*o43*o61*np.exp(o38+o71)-o109*o22*o29*o35*np.exp(o14+o27+o91)-2.*o22*o29*o5*o99*np.exp(o27+o91+o92+o93)-2.*o20*o22*o5*o99*np.exp(o91+o92+o93+2.*o4*o6*x)+o109*o22*o29*o43*np.exp(o38+o4*o6*(o63+xa))))/((o15*(o17*o20*o24-o28*o29*o31)*o35+(-(o24*o28*o29)+o17*o20*o31)*o39*o43)*o5*np.sqrt(o12))
    elif x <= xb[1]:
        return (1.*np.exp(-(o11*o13*x))*(-(o20*o24*o35*o6*o96*np.exp(o140+o16))-o20*o31*o43*o6*o96*np.exp(o149+o16)+o20*o31*o6*o99*np.exp(o16+o171)+o109*o18*o20*o35*np.exp(o16+o184)+o29*o31*o35*o6*o96*np.exp(o140+o27)+o24*o29*o43*o6*o96*np.exp(o149+o27)+o179*o24*o29*o6*np.exp(o171+o27)+o18*o195*o29*o35*np.exp(o184+o27)+o29*o31*o6*o99*np.exp(o27+o11*o13*(o37+o48))+o109*o18*o20*o43*np.exp(o16+o38+o92)+2.*o12*o160*o18*o43*np.exp(o38+o55+o91+o92)+o179*o20*o24*o6*np.exp(o14+o16+o93)+2.*o12*o160*o18*o35*np.exp(o55+o91+o92+2.*o11*o13*x)+o18*o195*o29*o43*np.exp(o27+o11*o13*(o147+xa))))/(o12*(o15*(-(o17*o20*o24)+o28*o29*o31)*o35+(o24*o28*o29-o17*o20*o31)*o39*o43)*np.sqrt(o5))
    else:
        return 0.
    
if __name__ == '__main__':
    num = 301
    xb = [0., 3.]
    xvals = np.linspace(xb[0], xb[1], num)
    xa = 1.0
    st = [1., 10.]
    sa = [0.01, 0.001]
    df = [1. / (3. * st[0]), 1./ (3. * st[1])]
    q = [1., 1.]
    j = [0.25, 0.]

    sol = np.zeros(num)

    for i in range(num):
        sol[i] = diffusion_solution(xvals[i],
                                    xb,
                                    xa,
                                    df,
                                    sa,
                                    q,
                                    j)
    
    plt.plot(xvals, sol)
    plt.show()
        
