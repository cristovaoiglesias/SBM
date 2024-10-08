using CSV, DataFrames
using Printf
using LinearAlgebra, DifferentialEquations
# using Plots; gr();
using Plots,Noise
using StatsPlots
using Dates
using Distributions
using Turing, Measures
# using  OptimizationOptimJL, OptimizationFlux
# using  ModelingToolkit , Optimization
# using  Optimization, OptimizationBBO
# using Symbolics
using LaTeXStrings
using MCMCChains
using Random
Random.seed!(1);
using Statistics
using StatsBase

FULL_PATH="" #e.g. "/Users/Thesis"

# ODE system for mAb production used in "Bioprocess optimization under uncertainty using ensemble modeling (2017)"
function mAb_production!(du, u, p, t)
    Xv, Xt, GLC, GLN, LAC, AMM, MAb = u
    mu_max, Kglc, Kgln, KIlac, KIamm, mu_dmax, Kdamm, Klysis, Yxglc, mglc, Yxgln, alpha1, alpha2, Kdgln, Ylacglc, Yammgln, r1, r2 ,lambda = p
    # mu_max  = 5.8e-2;
    # Kglc    = 0.75;
    # Kgln    = 0.075;
    # KIlac   = 171.756;
    # KIamm   = 28.484;
    # mu_dmax = 3e-2;
    # Kdamm   = 1.76;
    # Klysis  = 0.05511;
    # Yxglc   = 1.061e8;
    # mglc    = 4.853e-14;
    # Yxgln   = 5.57e8;
    # alpha1  = 3.4e-13;
    # alpha2  = 4;
    # Kdgln   = 9.6e-3;
    # Ylacglc = 1.399;
    # Yammgln = 4.27e-1;
    # r1 = 0.1;
    # r2 = 2;
    # lambda = 7.21e-9;

    mu = mu_max*(GLC/(Kglc+GLC))*(GLN/(Kgln+GLN))*(KIlac/(KIlac+LAC))*(KIamm/(KIamm+AMM));
    mu_d = mu_dmax/(1+(Kdamm/AMM)^2);

    du[1] = mu*Xv-mu_d*Xv;  #viable cell density XV
    du[2] = mu*Xv-Klysis*(Xt-Xv); #total cell density Xt
    du[3] = -(mu/Yxglc+mglc)*Xv;
    du[4] = -(mu/Yxgln+alpha1*GLN/(alpha2+GLN))*Xv - Kdgln*GLN;
    du[5] = Ylacglc*(mu/Yxglc+mglc)*Xv;
    du[6] = Yammgln*(mu/Yxgln+alpha1*GLN/(alpha2+GLN))*Xv+Kdgln*GLN;
    du[7] = (r2-r1*mu)*lambda*Xv;

end

u0 = [2e8    ,   2e8 ,   29.1, 4.9, 0.0, 0.310, 80.6] #initial condition from "Bioprocess optimization under uncertainty using ensemble modeling(2017)"
tstart=0.0
tend=103.0
# sampling= .125 #7.5 min
# sampling= 0.0625  #3.75
sampling= 1.0 #1 h
tgrid=tstart:sampling:tend

#RUN A
#parameters from the paper "Bioprocess optimization under uncertainty using ensemble modeling (2017)" with low mAb titer.
p = [5.8e-2, 0.75, 0.075, 171.756, 28.484, 3e-2, 1.76, 0.05511, 1.061e8, 4.853e-14, 5.57e8, 3.4e-13, 4, 9.6e-3, 1.399, 4.27e-1, 0.1, 2, 7.21e-9 ]
prob =  ODEProblem(mAb_production!, u0, (tstart,tend), p)
sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)

# Creating a NEW dataset with noise
# noise_dataset=add_gauss(transpose(sol),0.0)
# noise_dataset[:,1]=add_gauss(transpose(sol)[:,1],10e7) #specific noise used for Xv
# noise_dataset[:,2]=add_gauss(transpose(sol)[:,2],10e7)#specific noise
# noise_dataset[:,3]=add_gauss(transpose(sol)[:,3],1)
# noise_dataset[:,4]=add_gauss(transpose(sol)[:,4],.5)
# noise_dataset[:,5]=add_gauss(transpose(sol)[:,5],2.)
# noise_dataset[:,6]=add_gauss(transpose(sol)[:,6],.1)
# noise_dataset[:,7]=add_gauss(transpose(sol)[:,7],40.5)#specific noise
# noise_dataset=transpose(noise_dataset)

noise_dataset=[2.622709882387273e8 2.0194279199307412e8 1.6167862119212228e8 3.276807738059374e8 3.7935371642829883e8 1.9770802422624606e8 2.1191326223125517e8 2.4603471819068947e8 1.911783039082254e8 2.2429212546780646e8 3.2337844113259435e8 1.7912152440434566e8 3.0621404853213334e8 3.23901914684908e8 4.481615330406361e8 3.2453631613262945e8 3.995372389597404e8 5.231141802128235e8 5.8285037674442e8 4.6858072833689153e8 5.1962459494106376e8 4.861737138903069e8 6.244432826000214e8 4.4065169090418935e8 5.986214674397179e8 6.766616672004606e8 3.709646472477536e8 7.532846455931472e8 6.8181755413375e8 5.765508975766034e8 6.851650168811543e8 9.141156331501462e8 8.627252147288188e8 7.363368849504142e8 8.346649889467467e8 8.785722081890684e8 8.067671081280519e8 1.0180572442968919e9 9.205809509686451e8 8.603079776096922e8 9.795287820056833e8 1.0677966596343756e9 9.81462034397675e8 1.1920366655360231e9 1.1670020484540496e9 1.2323557234927135e9 1.2815469560125215e9 1.1991743823479967e9 1.249561070686404e9 1.3886995160436933e9 1.1796457689233913e9 1.2561466060974743e9 1.3020708597075052e9 1.3466064891768904e9 1.2290585881126304e9 1.2855600371265068e9 1.2552254262596998e9 1.5111157381179624e9 1.2650937759414754e9 1.3781036940813951e9 1.2314702667341425e9 1.1788340630956252e9 1.1345903346993806e9 1.2138756163002603e9 9.801906769518042e8 1.3761874346243305e9 1.154110435366774e9 1.03933307852024e9 9.152240606004863e8 1.1464449137679555e9 9.446681425145663e8 7.586967483857651e8 1.0147484064429291e9 1.02952408589797e9 1.0244610289589368e9 8.724187266558887e8 7.904463927813615e8 1.0360404494309562e9 9.302841755198325e8 7.935851289654316e8 8.442717983431426e8 7.930798705617493e8 7.967548342997178e8 7.184073667461457e8 7.817281196851547e8 6.819940907529664e8 6.443006212254937e8 6.030236857347547e8 5.828979407285898e8 8.241892464505432e8 7.506408983662498e8 5.50104427085231e8 6.658852375904856e8 6.551270026440921e8 5.2086278205869615e8 5.476629132298094e8 7.458493861149327e8 5.543397358500638e8 5.873736444261022e8 5.017270328448467e8 6.73445795764657e8 4.407982533762787e8 4.7703433059004384e8 5.1940485771500313e8; 2.1428819487901792e8 1.8372433151140982e8 3.737910616882963e8 1.782832293760267e8 3.6220385739934915e8 3.1419683289308566e8 4.542290842456165e8 4.436200788745836e8 1.5089871269999748e8 3.681445177139012e8 2.2789030391479635e8 3.1853466127063155e8 3.0179667278896546e8 5.726090159759759e8 3.993116783124364e8 3.766461074754633e8 3.878380216999912e8 4.263856677706928e8 3.593613590513307e8 5.082409061231533e8 4.420531307462955e8 5.56441291892079e8 5.967399776948133e8 5.643365297090917e8 7.745698721466902e8 6.875638873505797e8 6.673453151314206e8 5.203297018858541e8 6.872902752943093e8 7.62334370350743e8 6.856638652522278e8 7.248732220281342e8 9.281993797840326e8 9.202881948323021e8 9.537509522227796e8 1.0392754575223051e9 9.73746663989914e8 1.1068974018421142e9 9.477224677529842e8 1.1836422466716912e9 1.0637495400174644e9 1.1589314663276315e9 1.0652811356153684e9 1.211123403642006e9 1.3596512574505582e9 1.3745173906270013e9 1.3102394919110837e9 1.210679032615285e9 1.4004380684024396e9 1.5355367802789166e9 1.6514990018630588e9 1.6002682702361248e9 1.6382242082094622e9 1.639149346744396e9 1.6276609054027572e9 1.7716378154954658e9 1.6387636129476333e9 1.6439576736811554e9 1.6562457462924595e9 1.5951922843194017e9 1.68567675288444e9 1.737053251124392e9 1.6810230324578032e9 1.7515778964227867e9 1.5035273687464921e9 1.5962164244996865e9 1.4864163206352632e9 1.55182787388758e9 1.4002041148039637e9 1.5874862503886034e9 1.4794831837006693e9 1.3239143253534806e9 1.3371480502755296e9 1.3153480227161775e9 1.4116712573391113e9 1.0969704736637514e9 1.4144509782012947e9 1.2721350746430557e9 1.312892982181875e9 1.2568652429093337e9 1.179029147792027e9 1.1647482871651309e9 1.0405129589277313e9 1.2343576101991668e9 9.46948717786076e8 1.2148126268061314e9 1.135110019612623e9 1.2495066351735008e9 1.0080519938721552e9 1.0476459464002253e9 1.0969067439137506e9 8.346119560925379e8 9.361967502357726e8 9.156329028338491e8 8.899891893320222e8 8.797252198082141e8 7.579021771825961e8 8.719527065287284e8 8.477447708947272e8 9.021593049800278e8 8.117432305487416e8 7.626734895934641e8 8.335505499721094e8 8.32494286839952e8; 29.20277026467487 29.477294016173243 28.21136533085745 26.38632914178598 27.119271375526576 27.643571762919308 28.241335867321837 28.81220732920756 29.881534646210394 28.00663417619523 29.09436406801536 27.28395055962792 26.989455521506937 26.801287333502657 25.80994258736127 25.73434031065365 26.424107033727182 26.4276171980567 24.262557440866477 26.557581701658393 27.535032794135134 26.948786085337893 24.573786594819094 24.43816293357499 25.971886897460394 24.55654220222536 22.757632958110687 23.48747034401116 22.980383698601702 23.289630129849584 24.287185037368882 22.95032121616785 21.752457372057325 21.5483797310853 21.92840502857233 20.700518961894584 21.38259100727973 20.357710303002996 20.61089640280984 19.889216521914275 18.42601407070489 20.498728790224394 19.003672049955405 18.93727287664149 15.910204147824027 15.594844330584017 16.535239612677994 17.03695099070657 15.244473337577867 15.457005495717572 14.345705895403757 12.355697775024026 16.9526043049757 11.12461239401616 10.931534495648298 10.840439256819648 12.943903906054679 9.164862389117213 12.673889601610894 9.467850943858586 10.223999224178591 9.860384630790968 11.11755734569875 12.037396403159336 11.408918058572148 10.650726582006497 9.501068887636743 11.882558130355646 9.08667882107524 10.607850414988297 10.60656804779511 9.80815007631465 9.658696691028492 11.067867936679566 9.426114120856141 11.0306097781526 12.56103070578413 10.470290535105832 11.286138401919178 9.687108686580828 11.397376295471057 9.58644240716913 10.689087193401043 12.473133851394055 9.692609984933718 11.613905260674093 10.38565086969745 10.761442636777298 9.941933664863486 11.006346588521073 12.106533807816575 11.426514541740653 10.441163988975493 11.686075273208521 11.414215802176894 10.44952336755459 11.341378916818766 10.071013068505838 10.22603309896004 10.219143538835093 10.749916189537705 11.565999475969816 11.5468953372948 10.769645250371818; 4.572306405051611 4.75947737996425 4.8833740277223425 4.298826009235349 4.168923562109967 5.0105768158662265 4.723607326202752 4.017351744361896 4.1341806977497 3.296067929262998 4.565350452782476 4.560801956382956 4.757469841044863 3.312859625071665 3.656397131769241 3.8839312309124523 3.4239439325538217 3.6067502487534036 3.7243501836262562 3.480970305095409 3.148308623895112 1.7968631650502027 3.6377252783116782 2.623966042324539 2.493800256466182 2.843380737103265 3.1000462131062734 2.6666921346819397 3.124999014046541 2.7540399177203625 2.486298315705401 2.2970985885238213 1.6872621846665816 2.1476033468300084 1.9330540951433186 2.329442248267433 2.0433466610180346 3.084456070007492 1.7338431651620252 1.7502287982684288 1.5094462274525926 1.6767407724203367 1.595008660037029 0.5066397288323892 1.3125457512941952 1.3586230807976756 0.6040404653184277 0.8681928026152348 1.8165974169983388 0.8465842817877771 1.2965318855838621 1.2401053085682936 1.6187147497317833 -0.1828377160955612 0.03179530153618404 0.29152381232712443 -0.2112016581271375 0.42841074223788533 -1.1571118754972995 -1.0193445294214019 -0.3830920589875316 -0.40437536242871897 0.967150273776057 0.8278487263611076 0.44968293278926125 0.6665985648412674 0.3590778788464661 -0.39959156994829703 -0.10476743705722334 -0.47606465531997016 0.25104686633812323 0.4814313752907306 -0.4545439847879182 -0.6839477400118419 -0.4306411705162389 0.6337645870332279 0.5933484985513895 0.8452798447245613 0.8228013467913725 -0.2468527082764849 0.08616319325275383 0.09982930190722672 -0.8014753242119892 0.09946040287067172 0.4537957975342864 0.7396067326025452 -0.35394442945054305 -0.4485442727736141 -0.15698430574590605 0.10443778151450508 -0.5657949508423331 -0.3522196547364423 -0.0801957682753377 0.18726754438504276 0.9553925669402515 0.05168693374112474 0.1849555745059573 -0.9752061328360352 0.6698241896955822 -0.27333573484040424 0.48513725442025174 0.0824560785355206 0.21209655376798217 -0.2506040182871947; -2.8917636470149763 -1.9223439095389163 -1.7237203791743667 3.75052028510859 -3.586579107998009 -0.7283735034245234 1.365458751697759 -0.44963796623711727 1.257422585788416 -2.4627621905612562 5.238521846533116 -1.3777000783465487 -0.953565112209724 -0.0847259085155625 2.9741448451910344 5.53692854038775 4.105846602868125 6.67363445441803 -0.47952179235975567 4.3422745573663795 7.16447623850906 5.804800905096104 5.199323415238607 4.969217461440027 5.590596412794953 5.332359661650347 6.1600132951910584 7.50768359575427 5.252141745537713 10.227165364503584 9.440641615900073 9.11993736861247 7.88598492782977 7.334926659680082 10.953309835979624 9.872776678898582 8.604473464257378 14.938527704415186 14.909999288601703 12.136615963885303 12.886119154617285 18.048672113366344 15.477555080076941 15.394726654331626 18.939530386199095 12.411734155358097 18.158194030332865 18.889748163660744 15.968064122252489 20.040717661221926 19.89147109380299 19.500377834962755 19.511390378948185 23.186092278708273 19.460430803777903 27.323430646054838 25.504024870253275 23.772381110864853 28.018452646895113 20.52343836349276 26.03158227308219 25.73399690786045 22.10241428567713 23.134361871046142 29.582956602925808 23.731958558806497 23.684540931651675 24.462181571516474 25.846063114469466 25.093026805053864 25.67211720223557 22.824447962892062 26.150425804545034 25.894764340439792 25.990808976290968 22.664993438067487 23.870368476492345 24.854400236279762 25.73808788341794 23.09874399717675 27.039577304248926 26.248680484987066 22.009203964592444 25.29095932237809 19.478906595668175 22.840048083389583 24.27955385006397 26.164516821157704 25.48494831611489 26.249308201170436 25.742764240795317 26.85387259202552 22.524871429244868 25.129447635185652 24.10373230085818 23.692759649912688 22.9258702864297 24.012938758345424 22.37777252861951 27.05627178461125 24.716862940316187 25.503632416644116 25.77202710834998 21.932708805723387; 0.25722460829921456 0.33498915123775597 0.4027836203177324 0.4691651850307952 0.5834812804019149 0.5352570256946412 0.9076805681305348 0.6401930595469243 0.9498987698989476 0.7386022110451885 0.8303342346605279 0.8097940754358821 0.9458701910383305 1.2026589427983705 1.0874502859878177 1.1342087522218576 1.1597620794396717 1.1165663468999247 1.118865692890915 1.4454994651182342 1.3791861524791802 1.4821180702742431 1.4095088789259655 1.6531971511011794 1.5676780195252344 1.7298646687167065 1.598761495665919 1.7224438723727509 1.9406074573036909 1.8927043683847402 1.7940054960538532 1.8571211094237792 2.1047140060067755 2.187138856976316 2.1041818767860048 1.8746537507034295 2.270225802295759 2.222166555633762 2.25402418026911 2.3124344063230002 2.6185775200476473 2.245620975917343 2.470767650580562 2.643924277414102 2.5971704534211066 2.607076994799372 2.622685432547014 2.762551841918591 2.8309471125407675 2.7902985995790535 2.8021633392540903 2.936905074652936 3.082776405796304 3.117229834848234 2.8800445502235914 3.196759200835972 3.208996609829755 3.2782110262064235 3.2814174546036607 3.1136407406226407 3.180139267853858 3.229676255539479 3.3256403191175825 3.2092293455155376 3.3977156676946887 3.1958702843468445 3.2871102736896898 3.1576901160268025 3.098217217232399 3.2556652048852595 3.295489654924931 3.2053184474028145 3.1928533704638116 3.0240907347199566 3.3477573332792683 3.2685201589821076 3.2362429624387703 3.2274513302280927 3.0604177096091383 3.2425211722641705 3.1964792204773897 2.996515061470883 2.9666524158758367 3.199632001931098 3.4184323470615166 3.087803651630721 3.421553547704577 3.4414529230467 3.3520396364765923 3.3003204027231083 3.1476920765614405 3.1983434308882193 3.125823171112305 3.328222945285563 3.2401340072726525 3.1570169516437856 3.1572037094861387 3.309635215665876 3.2319714855840056 3.37930515949849 3.118941894847676 3.321040991493803 3.1575371475088234 3.3330491950734102; 21.27600314809022 45.44065989653222 102.81408109206126 121.7133571100751 126.60738048774137 68.16041318501269 97.87234028658567 146.0279712406297 190.34198622801213 104.24046942857959 168.82256924990108 157.96909257555157 116.35920751643208 154.994908625421 158.5806532803913 101.9529908737392 114.40676436200026 181.24629282210418 139.88715051909674 101.0716289574107 124.38854824214512 183.67274395874728 202.4722226440912 213.78403236699415 258.3575379915837 180.91754716665974 187.4684392775224 226.5729590336052 219.91884975946692 255.49259396512483 258.56305063929165 263.66704894001333 220.35554238433335 326.34732934994224 357.20728996534024 247.63075221299795 395.2876840059106 297.7434357940236 359.87272166102883 333.16207130833504 387.1325817800238 427.0006281479464 419.14156787331143 399.61535637383463 538.518154628835 542.1641695332054 466.73523348495075 436.5075624064259 477.74231739578073 509.09058735496467 573.0918167081153 525.5651684562987 505.29705512228816 567.4358336543971 559.8945200671025 679.5116190912001 605.2854957601689 738.5379329569663 728.9130534733492 718.5453474635716 749.1285647214121 851.1017555274574 793.2118259110408 794.5375955736874 804.2839507246968 854.0942742191095 845.8586088814296 914.5906329168736 871.1015383326073 826.0161095090772 943.341663233339 978.1398840860968 861.8688126741878 957.762719971393 965.6730122642296 939.4088520447183 965.5541888960452 1006.3263528327203 999.4600504074475 1024.7101411085685 1067.7381722244547 1036.0104787117173 1018.662764676286 1070.6698717202837 1044.7741516412955 1078.4949598460787 1079.351766620866 1158.0909039298983 1091.4238596884416 1159.262546606043 1125.5769221729424 1177.1385624803788 1172.3618838689665 1210.2376209328852 1169.483996937815 1185.88069195908 1271.7986487967657 1247.0859072981918 1192.5694958386684 1236.6657093917363 1209.892770785825 1287.125116623224 1298.5454055840344 1208.8423544510729]

plots=
scatter(sol.t,transpose(noise_dataset), size=(1000,800),color=:lightblue, lw=1.5, label = "Noise", ylabel=["[Xv]" "[Xt]"  "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,3))
plot!(sol.t,transpose(sol),color=[:red :red :red :red :red :red :red], lw=3.5, label = "True", ylabel=["[Xv]" "[Xt]"  "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,3))
display(plots)




true_params= [5.8e-2,0.05511,9.6e-3,1.399,4.27e-1,7.21e-9]

#
# estimated_params=[0.0579140814698154, 0.05796936920630466, 0.009817355250782624, 1.3643988589985205, 0.39985101473644247, 7.184223166953116e-9]
# v_min=BBF1(estimated_params)
# # v_min=BBF1(minimizer(estimated_params))
# println("v_min from PS :",v_min)
# Vmin_dist = rand(truncated(Normal(v_min,v_min*0.1),0,1e40), 104)
# println("Vmin_dist :",Vmin_dist)

Vmin_dist =[64083.303686825486, 75896.70565822984, 60726.61806134732, 60670.916206499154, 67799.82483092786, 72907.28756465545, 78542.7145764476, 65817.4083033723, 76054.0298559056, 50337.77974143732, 59778.19948531619, 63069.88712385831, 60627.52951397017, 59607.01685339256, 66275.61299029489, 50899.98638949888, 64082.74417307006, 59345.12082239471, 56024.714678284465, 64773.26991934439, 54668.4704307053, 64259.448209633476, 56855.2954680986, 60206.35879985898, 74774.1389094797, 63502.827489010684, 65492.99289097107, 61918.494540937354, 58939.165815141205, 64341.80698648235, 66415.0909765221, 64722.4892669979, 60878.71380245699, 47810.40300561873, 67644.54657891818, 57193.50576032336, 70861.66501290811, 68457.61134531509, 71423.24451772383, 71789.1726826851, 61999.98696951531, 67920.01360724343, 58828.94310292536, 67003.59038972453, 61095.990112132, 68100.99747576904, 55240.019443914825, 63118.00316149414, 64931.18580963496, 63558.3015672119, 64252.939301614504, 60505.42643626012, 66694.46260015616, 56352.94803584852, 53421.71209831399, 66832.07817226047, 66592.16185508925, 63756.839524948184, 70906.0994552915, 69222.87248174845, 55542.53525117473, 66730.66443650378, 61056.53321946482, 56555.98734891488, 71723.36584536413, 66513.45921221032, 63348.96269557391, 67880.14027545076, 66336.21324943313, 61399.58641858746, 80864.72024620048, 71915.06771024238, 53391.069227467255, 75114.89130183255, 68861.35902190165, 65347.91049833318, 70945.77541843704, 70094.51609280214, 71448.41642931252, 62506.91019809046, 56975.412113334605, 64170.62353558737, 55652.929764821296, 58595.688862288356, 68949.55464672975, 67559.74197541943, 60606.22669510783, 61662.64562026772, 74195.68381511763, 52839.28544261155, 58459.67553110174, 69616.50632344109, 78068.48572089437, 53563.867913576716, 62491.20335887128, 61422.59363236653, 61318.71222017127, 69597.84589100452, 61056.39740469732, 75918.07044452532, 59004.88253313118, 60587.77226002062, 50716.889913133724, 65537.79847868643]

# Vmin_dist_wide =rand(truncated(Normal(v_min,v_min*0.75),0,1e40), 104)
Vmin_dist_wide =[67210.6305222711, 120170.379624127, 131183.43121735213, 135318.25936687278, 28361.149803253058, 150333.8887069102, 67229.09535329633, 37498.57363610351, 26416.45967573604, 66296.70195524648, 52233.25196963979, 129576.73530562066, 55341.40894354868, 136622.87289081363, 51315.348023486615, 96042.25701805518, 125957.68258407348, 181362.23975486917, 55845.857244065264, 155980.21354200412, 86492.73851192919, 81044.76540138226, 88296.46056679783, 79610.64496652219, 95770.2640091679, 75681.84549326039, 18837.69311547003, 47446.887457882025, 97512.89067493012, 47397.678266482006, 27202.13445689965, 68582.22303016899, 116850.41798301094, 16677.629892752724, 88026.179689925, 68460.40534776798, 65122.089174019304, 75856.213231263, 8393.464301492582, 106297.15986445674, 102618.0911831557, 50105.334315418026, 3946.6301142095763, 91198.1124821347, 100660.60488820873, 63318.74991409761, 152660.9261790458, 39476.93842839373, 57686.74168538033, 61161.9762679878, 127626.82459961882, 52928.58300853899, 72474.50014966496, 42268.56285395416, 145083.8748282928, 161751.1767843645, 4057.6972526151003, 79538.34806333415, 69461.07508146687, 47235.12504708493, 97982.27522794149, 74734.7924947308, 35617.202780463245, 56893.60220829651, 104409.71239993203, 34513.75642046189, 50317.2287949, 35017.72280986587, 51831.22314530825, 136033.04474868305, 91749.58949051207, 37453.719938957554, 76599.67713461304, 37061.95375216437, 161129.25825262468, 64725.60654781828, 100867.1283979859, 114029.01849060929, 64276.12059423946, 32595.712702860143, 97475.92939915473, 62866.57132354602, 120002.55039353753, 75460.08999731469, 29603.178146332146, 85338.35803707642, 41224.85956733307, 28930.901818732018, 44538.09210540487, 66983.33743059897, 48261.92291062272, 65646.34747173011, 66256.9795763275, 33456.35319694456, 74680.39682109455, 177.29255098364956, 123800.62336495305, 64204.13949365799, 167428.35807969212, 73055.92351363064, 19720.126812565897, 56375.14299942203, 87953.4037903228, 145289.39455555734]

# Create the density plot
density_plot = density(Vmin_dist, title="Surrugate distribution", xlabel="Values", ylabel="Density")
density!(Vmin_dist_wide)
display(density_plot)






# load all results of exp4 folder
directory =  FULL_PATH*"/SBM/empirical_tests/task1/step5/set100"
all_files = readdir(directory)
only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)
casd=1

all_chains_bf=[]
for i in txt_files
    if occursin("classical_bf_",i )
        println(casd ,"| filename: ",i," | time: ",parse(Float64, split(i, "_time_")[2][1:9]))
        global casd=casd+1
        ch = CSV.read(directory*"/"*i, DataFrame)
        append!(all_chains_bf,[ch])
    end
end





#removing outliers (failed runs) from set
all_chains_bf[4]=all_chains_bf[5]
all_chains_bf[22]=all_chains_bf[23]
all_chains_bf[29]=all_chains_bf[28]
all_chains_bf[32]=all_chains_bf[31]
all_chains_bf[38]=all_chains_bf[37]
all_chains_bf[41]=all_chains_bf[42]
all_chains_bf[54]=all_chains_bf[52]
all_chains_bf[58]=all_chains_bf[57]
all_chains_bf[60]=all_chains_bf[61]
all_chains_bf[64]=all_chains_bf[63]
all_chains_bf[70]=all_chains_bf[71]
all_chains_bf[74]=all_chains_bf[76]
all_chains_bf[79]=all_chains_bf[75]
all_chains_bf[84]=all_chains_bf[89]
all_chains_bf[85]=all_chains_bf[82]
all_chains_bf[87]=all_chains_bf[90]
all_chains_bf[88]=all_chains_bf[93]
all_chains_bf[91]=all_chains_bf[97]
all_chains_bf[92]=all_chains_bf[98]

# 4classical_bf_12_time_823.2544419765472.csv
# 22classical_bf_29_time_818.4260578155518.csv
# 29classical_bf_35_time_5594.879766941071.csv
# 32classical_bf_38_time_298.14608097076416.csv
# 38classical_bf_43_time_757.2057549953461.csv
# 41classical_bf_46_time_5879.856750011444.csv
# 54classical_bf_58_time_1175.865051984787.csv
# 58classical_bf_61_time_5534.256143093109.csv
# 60classical_bf_63_time_955.6755590438843.csv
# 64classical_bf_67_time_604.0163490772247.csv
# 70classical_bf_72_time_178.17166590690613.csv
# 74classical_bf_76_time_5619.2471652030945.csv
# 79classical_bf_80_time_1034.3357501029968.csv
# 84classical_bf_85_time_5609.64151597023.csv
# 85classical_bf_86_time_6185.11092710495.csv
# 87classical_bf_88_time_294.62730407714844.csv
# 88classical_bf_89_time_630.19824385643.csv
# 91classical_bf_91_time_650.6769709587097.csv
# 92classical_bf_92_time_387.0757191181183.csv



# load all results of exp2 with surrugate dist size 4 folder
directory =  FULL_PATH*"/SBM/empirical_tests/task1/step3/set100"
all_files = readdir(directory)
only_files = filter(f -> isfile(joinpath(directory, f)), all_files)
txt_files = filter(f -> endswith(f, ".csv") && isfile(joinpath(directory, f)), all_files)

all_chains_bbf1=[]
casd=1
for i in txt_files
    if occursin("chain_bbf1_4_",i )
        println(i)
        ch = CSV.read(directory*"/"*i, DataFrame)
        append!(all_chains_bbf1,[ch])
    end
end






function credible_interval2(dist)
# Calculate the credible interval (e.g., 95%)
    # mean_v = mean(dist)
    # std_v = std(dist)
    # ci_lower = quantile(Normal(mean_v, std_v), 0.025)
    # ci_upper = quantile(Normal(mean_v, std_v), 0.975)
    # return [ci_lower,ci_upper]
    return quantile(dist, [0.025, 0.975])
end


all_ci_bf=[]
for i in 1:100
    civ=[credible_interval2(all_chains_bf[i][:,3]),
        credible_interval2(all_chains_bf[i][:,4]),
        credible_interval2(all_chains_bf[i][:,5]),
        credible_interval2(all_chains_bf[i][:,6]),
        credible_interval2(all_chains_bf[i][:,7]),
        credible_interval2(all_chains_bf[i][:,8])]
    append!(all_ci_bf,[civ])
end

all_ci_bbf1=[]
for i in 1:101
    civ=[credible_interval2(all_chains_bbf1[i][:,3]),
        credible_interval2(all_chains_bbf1[i][:,4]),
        credible_interval2(all_chains_bbf1[i][:,5]),
        credible_interval2(all_chains_bbf1[i][:,6]),
        credible_interval2(all_chains_bbf1[i][:,7]),
        credible_interval2(all_chains_bbf1[i][:,8])]
    append!(all_ci_bbf1,[civ])
end




# purple2 navyblue royalblue1




##
# MEAN OF all credible interval
#





corbbf1=:purple2
corbbf1std104=:royalblue1


ll=[L"Density~\mu_{max}" L"Density~K_{lysis}" L"Density~K_{dgln}" L"Density~Y_{lac,glc}" L"Density~Y_{amm,gln}" L"Density~\lambda"]
true_params= [5.8e-2,0.05511,9.6e-3,1.399,4.27e-1,7.21e-9]
#plot to answer RQ4
va=0.05
density_plot1 = density(Array(all_chains_bbf1[1][:,3:8]),  alpha = va, c=corbbf1std104, lenged=false ,xlabel="", ylabel=ll, layout=(6,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bbf1[i][:,3:8]),  c=corbbf1std104,alpha = va, legend=false)
end
for (i, true_val) in enumerate(true_params)
    # for ci95 in hcat(all_ci_bbf1...)'[:,i]
    #     #mean of all credible interval 95
    #     vline!(ci95, linestyle=:dash, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=i)
    # end
    #mean of all credible interval 95
    vline!(mean(hcat(all_ci_bbf1...)'[:,i]), linestyle=:dash, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=i)
    vline!(density_plot1, [true_val], lw=2, lc=:red, layout=(6,1), subplot=i)

end
# title!(density_plot1, L"SBM-BBF1~(4~\nu_{min}~values)", subplot=1,titlefontsize=12)
title!(density_plot1, L"SBM-BBF1~(NSDS4)", subplot=1,titlefontsize=12)
xticks!(density_plot1, [0.05, 0.06, 0.07], subplot=1)
xticks!(density_plot1, [0.0045, 0.01475, 0.025, 0.045], subplot=3)
xticks!(density_plot1, [0,1e-8, 2.5e-8], subplot=6)
xlabel!(density_plot1, "Sample values", subplot=6,labelfontsize=10)
# plot!(density_plot1,[NaN], label="95% CI", color=:red,lw=lws,  linestyle=:dash, subplot=1)
# vline!(density_plot1,[NaN], lengend=true, linestyle=:dash, color=corbbf1std104, label="95% CI", subplot=1)


density_plot2 = density(Array(all_chains_bf[1][:,3:8]),  alpha = va, c=corbbf1, lenged=false ,xlabel="", ylabel="", layout=(6,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bf[i][:,3:8]),  c=corbbf1,alpha = va, legend=false)
end
for (i, true_val) in enumerate(true_params)
    # for ci95 in hcat(all_ci_bf...)'[:,i]
    #mean of all credible interval 95
        # vline!(ci95, linestyle=:dash, color=corbbf1, label="95% CI",layout=(6,1), subplot=i)
    # end
    #mean of all credible interval 95
    vline!(mean(hcat(all_ci_bf...)'[:,i]), linestyle=:dash, color=corbbf1, label="95% CI",layout=(6,1), subplot=i)
    vline!(density_plot2, [true_val], lw=2, lc=:red, layout=(6,1), subplot=i)
end

title!(density_plot2, L"CBM", subplot=1,titlefontsize=12)
xticks!(density_plot2, [0.057, 0.058, 0.059], subplot=1)
xticks!(density_plot2, [0.0089, 0.0094, 0.0099], subplot=3)
xticks!(density_plot2, [0.4, 0.42, 0.44, 0.46], subplot=5)
xticks!(density_plot2, [6.9e-9, 7.15e-9, 7.4e-9], subplot=6)
xlabel!(density_plot2, "Sample values", subplot=6,labelfontsize=10)

lws=3
pltL1 = vline([NaN], label="95% Credible Interval", color=corbbf1std104,lw=lws,  linestyle=:dot)
vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
plot!([NaN], label="Ground truth value", color=:red, lw=lws)

pltL2 = vline([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
plot!([NaN], label="Ground truth value", color=:red, lw=lws)

plot1l1 = plot(density_plot1,pltL1,layout=(grid(2,1, heights=[0.96, 0.04])), size=(600, 800))
plot2l2 = plot(density_plot2,pltL2, layout=(grid(2,1, heights=[0.96, 0.04])), size=(600, 800))

allplots = plot(plot1l1,plot2l2, layout=(1,2))
plot!(left_margin=4mm, bottom_margin=1mm,grid=false)
display(allplots)
# savefig(FULL_PATH*"/SBM/empirical_tests/results_analysis/rq4.png")





##
# all credible interval
#


corbbf1=:purple2
corbbf1std104=:royalblue1


ll=[L"Density~\mu_{max}" L"Density~K_{lysis}" L"Density~K_{dgln}" L"Density~Y_{lac,glc}" L"Density~Y_{amm,gln}" L"Density~\lambda"]
true_params= [5.8e-2,0.05511,9.6e-3,1.399,4.27e-1,7.21e-9]
#plot to answer RQ4
va=0.05
density_plot1 = density(Array(all_chains_bbf1[1][:,3:8]),  alpha = va, c=corbbf1std104, lenged=false ,xlabel="", ylabel=ll, layout=(6,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bbf1[i][:,3:8]),  c=corbbf1std104,alpha = va, legend=false)
end
for (i, true_val) in enumerate(true_params)
    for ci95 in hcat(all_ci_bbf1...)'[:,i]
        #mean of all credible interval 95
        vline!(ci95, linestyle=:dash,  alpha = va, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=i)
    end
    #mean of all credible interval 95
    # vline!(mean(hcat(all_ci_bbf1...)'[:,i]), linestyle=:dash, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=i)
    vline!(density_plot1, [true_val], lw=2, lc=:red, layout=(6,1), subplot=i)

end
# title!(density_plot1, L"SBM-BBF1~(4~\nu_{min}~values)", subplot=1,titlefontsize=12)
title!(density_plot1, L"SBM-BBF1~(NSDS4)", subplot=1,titlefontsize=12)
xticks!(density_plot1, [0.05, 0.06, 0.07], subplot=1)
xticks!(density_plot1, [0.02, 0.06, 0.08], subplot=2)
xticks!(density_plot1, [0.0045, 0.0185], subplot=3)
# xticks!(density_plot1, [0.0045, 0.01475, 0.025, 0.045], subplot=3)
xticks!(density_plot1, [0,1e-8, 2.5e-8], subplot=6)
yticks!(density_plot1, [0,2,4,6], subplot=4)
yticks!(density_plot1, [0,1e8,2.5e8], subplot=6)


xlabel!(density_plot1, "Sample values", subplot=6,labelfontsize=10)
# plot!(density_plot1,[NaN], label="95% CI", color=:red,lw=lws,  linestyle=:dash, subplot=1)
# vline!(density_plot1,[NaN], lengend=true, linestyle=:dash, color=corbbf1std104, label="95% CI", subplot=1)


density_plot2 = density(Array(all_chains_bf[1][:,3:8]),  alpha = va, c=corbbf1, lenged=false ,xlabel="", ylabel="", layout=(6,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bf[i][:,3:8]),  c=corbbf1,alpha = va, legend=false)
end
for (i, true_val) in enumerate(true_params)
    for ci95 in hcat(all_ci_bf...)'[:,i]
    #mean of all credible interval 95
        vline!(ci95, linestyle=:dash, color=corbbf1,  alpha = va, label="95% CI",layout=(6,1), subplot=i)
    end
    #mean of all credible interval 95
    # vline!(mean(hcat(all_ci_bf...)'[:,i]), linestyle=:dash, color=corbbf1, label="95% CI",layout=(6,1), subplot=i)
    vline!(density_plot2, [true_val], lw=2, lc=:red, layout=(6,1), subplot=i)
end

title!(density_plot2, L"CBM", subplot=1,titlefontsize=12)
xticks!(density_plot2, [0.057, 0.058, 0.059], subplot=1)
xticks!(density_plot2, [0.0089, 0.0099], subplot=3)
xticks!(density_plot2, [0.4, 0.42, 0.44, 0.46], subplot=5)
xticks!(density_plot2, [6.9e-9,  7.4e-9], subplot=6)
yticks!(density_plot2, [0,1000,2500], subplot=3)
yticks!(density_plot2, [0,20,40,60], subplot=5)

xlabel!(density_plot2, "Sample values", subplot=6,labelfontsize=10)

lws=3
pltL1 = vline([NaN], label="95% Credible Interval", color=corbbf1std104,lw=lws,  linestyle=:dot)
vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
plot!([NaN], label="Ground truth value", color=:red, lw=lws)

pltL2 = vline([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
plot!([NaN], label="Ground truth value", color=:red, lw=lws)

plot1l1 = plot(density_plot1,pltL1,layout=(grid(2,1, heights=[0.96, 0.04])), size=(600, 800),xtickfontsize=10,ytickfontsize=10,yguidefontsize=12)
plot2l2 = plot(density_plot2,pltL2, layout=(grid(2,1, heights=[0.96, 0.04])), size=(600, 800),xtickfontsize=10,ytickfontsize=10)

allplots = plot(plot1l1,plot2l2, layout=(1,2))
plot!(left_margin=5mm, bottom_margin=2mm,grid=false)
display(allplots)
savefig(FULL_PATH*"/SBM/empirical_tests/results_analysis/Figure_6.png")







##
# estimated noises SBM with variance and CBM with std
#

all_ci_bbf1=[]
for i in 1:101
    civ=[credible_interval2(all_chains_bbf1[i][:,9])]
    append!(all_ci_bbf1,[civ])
end
all_ci_bf=[]
for i in 1:100
    civ=[credible_interval2(all_chains_bf[i][:,9]),
        credible_interval2(all_chains_bf[i][:,10]),
        credible_interval2(all_chains_bf[i][:,11]),
        credible_interval2(all_chains_bf[i][:,12]),
        credible_interval2(all_chains_bf[i][:,13]),
        credible_interval2(all_chains_bf[i][:,14]),
        credible_interval2(all_chains_bf[i][:,15])]
    append!(all_ci_bf,[civ])
end
corbbf1=:purple2
corbbf1std104=:royalblue1


ll=[L"\sigma_{Xv}" L"\sigma_{Xt}" L"\sigma_{GLC}" L"\sigma_{GLN}" L"\sigma_{LAC}" L"\sigma_{AMM}" L"\sigma_{mAb}"]
true_params= [64083.3*0.1] #std value used to create the narrow surrogate model-> N(64083.3,64083.3*0.1)
#plot to answer RQ4
va=0.095
density_plot1 = density(Array(all_chains_bbf1[1][:,9:9]),  alpha = va, c=corbbf1std104, label=L"\sigma^2" ,xlabel="", ylabel=L"\sigma_{\nu}^2", layout=(7,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bbf1[i][:,9:9]),  c=corbbf1std104,alpha = va, label=L"\sigma^2")
end
vline!(density_plot1, true_params, lw=2, lc=:red, layout=(7,1), subplot=1)
#
# for (i, true_val) in enumerate(true_params)
for ci95 in hcat(all_ci_bbf1...)'[:,1]
    #mean of all credible interval 95
    vline!(ci95, linestyle=:dash,  alpha = 0.5, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=1)
end
#     #mean of all credible interval 95
#     # vline!(mean(hcat(all_ci_bbf1...)'[:,i]), linestyle=:dash, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=i)
#     vline!(density_plot1, [true_val], lw=2, lc=:red, layout=(7,1), subplot=i)
# end

# title!(density_plot1, L"SBM-BBF1~(4~\nu_{min}~values)", subplot=1,titlefontsize=12)
title!(density_plot1, "Density of SBM-BBF1(NSDS4)", subplot=1,titlefontsize=12)
xticks!(density_plot1, [1e3,25e3,5e4], subplot=1)
xaxis!(density_plot1, [1e3, 5e4],subplot=1)
yticks!(density_plot1, [0,0.00005,0.0001], subplot=1)

# yticks!(density_plot1, [0,1e8,2.5e8], subplot=6) , legend=L"\sigma^2"

# density([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
# plot!(density_plot1,legend=L"\sigma^2",subplot=1)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=2)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=3)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=4)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=5)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=6)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=7)


xlabel!(density_plot1, "Sample values",labelfontsize=10,subplot=1)
# plot!(density_plot1,[NaN], label="95% CI", color=:red,lw=lws,  linestyle=:dash, subplot=1)
# vline!(density_plot1,[NaN], lengend=true, linestyle=:dash, color=corbbf1std104, label="95% CI", subplot=1)


true_params=[1e7, 1e7, 1, 0.5, 0.1, 0.1, 40.5]#.^2
density_plot2 = density(Array(all_chains_bf[1][:,9:15]),  alpha = va, c=corbbf1, lenged=false ,xlabel="", ylabel=ll, layout=(7,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bf[i][:,9:15]),  c=corbbf1,alpha = va, legend=false)
end
for (i, true_val) in enumerate(true_params)
    for ci95 in hcat(all_ci_bf...)'[:,i]
        vline!(ci95, linestyle=:dash, color=corbbf1,  alpha = va, label="95% CI",layout=(6,1), subplot=i)
    end
    #mean of all credible interval 95
    # vline!(mean(hcat(all_ci_bf...)'[:,i]), linestyle=:dash, color=corbbf1, label="95% CI",layout=(6,1), subplot=i)
    vline!(density_plot2, [true_val], lw=2, lc=:red, layout=(7,1), subplot=i)
end

title!(density_plot2, "Densities of CBM", subplot=1,titlefontsize=12)
xticks!(density_plot2, [1e7,10e7,17.5e7], subplot=1)
xticks!(density_plot2, [1e7,10e7,17.5e7], subplot=2)
xticks!(density_plot2, [0.1,0.12,0.14,0.16], subplot=6)
yticks!(density_plot2, [1e-8,3e-8,6e-8,], subplot=1)
yticks!(density_plot2, [1e-8,3e-8,6e-8,], subplot=2)
yticks!(density_plot2, [0,3,6], subplot=3)
yticks!(density_plot2, [0,5,10], subplot=4)
yticks!(density_plot2, [0,20,40], subplot=6)
# xticks!(density_plot2, [0.0089, 0.0099], subplot=3)
# xticks!(density_plot2, [0.4, 0.42, 0.44, 0.46], subplot=5)
# xticks!(density_plot2, [6.9e-9,  7.4e-9], subplot=6)
# yticks!(density_plot2, [0,1000,2500], subplot=3)
# yticks!(density_plot2, [0,20,40,60], subplot=5)
#
xlabel!(density_plot2, "Sample values", subplot=7,labelfontsize=10)

lws=3
# pltL1 = vline([NaN], label="95% Credible Interval", color=corbbf1std104,lw=lws,  linestyle=:dot)
# vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
# plot!([NaN], label="Ground truth value", color=:red, lw=lws)
#
# pltL2 = vline([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
# vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
# plot!([NaN], label="Ground truth value", color=:red, lw=lws)

plot1l1 = plot(density_plot1, size=(600, 800),xtickfontsize=10,ytickfontsize=10,yguidefontsize=16, legend=false)
plot2l2 = plot(density_plot2, size=(600, 800),xtickfontsize=10,ytickfontsize=10, yguidefontsize=16, legend=false)

allplots = plot(plot1l1,plot2l2, layout=(1,2))
plot!(left_margin=5mm, bottom_margin=2mm,grid=false)
display(allplots)


# savefig(FULL_PATH*"/SBM/empirical_tests/results_analysis/estimated_sigmas_rq4.png")













##
# estimated noises SBM with variance and CBM with variance
#

all_ci_bbf1=[]
for i in 1:101
    civ=[credible_interval2(all_chains_bbf1[i][:,9])]
    append!(all_ci_bbf1,[civ])
end
all_ci_bf=[]
for i in 1:100
    civ=[credible_interval2(all_chains_bf[i][:,9].^2),
        credible_interval2(all_chains_bf[i][:,10].^2),
        credible_interval2(all_chains_bf[i][:,11].^2),
        credible_interval2(all_chains_bf[i][:,12].^2),
        credible_interval2(all_chains_bf[i][:,13].^2),
        credible_interval2(all_chains_bf[i][:,14].^2),
        credible_interval2(all_chains_bf[i][:,15].^2)]
    append!(all_ci_bf,[civ])
end
corbbf1=:purple2
corbbf1std104=:royalblue1


ll=[L"\sigma_{Xv}^2" L"\sigma_{Xt}^2" L"\sigma_{GLC}^2" L"\sigma_{GLN}^2" L"\sigma_{LAC}^2" L"\sigma_{AMM}^2" L"\sigma_{mAb}^2"]
true_params= [64083.3*0.1] #std value used to create the narrow surrogate model-> N(64083.3,64083.3*0.1)
#plot to answer RQ4
va=0.095
density_plot1 = density(Array(all_chains_bbf1[1][:,9:9]),  alpha = va, c=corbbf1std104, label=L"\sigma^2" ,xlabel="", ylabel=L"\sigma_{\nu}^2", layout=(7,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bbf1[i][:,9:9]),  c=corbbf1std104,alpha = va, label=L"\sigma^2")
end
vline!(density_plot1, true_params, lw=2, lc=:red, layout=(7,1), subplot=1)
#
# for (i, true_val) in enumerate(true_params)
for ci95 in hcat(all_ci_bbf1...)'[:,1]
    #mean of all credible interval 95
    vline!(ci95, linestyle=:dash,  alpha = 0.5, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=1)
end
#     #mean of all credible interval 95
#     # vline!(mean(hcat(all_ci_bbf1...)'[:,i]), linestyle=:dash, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=i)
#     vline!(density_plot1, [true_val], lw=2, lc=:red, layout=(7,1), subplot=i)
# end

# title!(density_plot1, L"SBM-BBF1~(4~\nu_{min}~values)", subplot=1,titlefontsize=12)
title!(density_plot1, "Density of SBM-BBF1(NSDS4)", subplot=1,titlefontsize=12)
xticks!(density_plot1, [1e3,25e3,5e4], subplot=1)
xaxis!(density_plot1, [1e3, 5e4],subplot=1)
yticks!(density_plot1, [0,0.00005,0.0001], subplot=1)

# yticks!(density_plot1, [0,1e8,2.5e8], subplot=6) , legend=L"\sigma^2"

# density([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
# plot!(density_plot1,legend=L"\sigma^2",subplot=1)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=2)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=3)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=4)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=5)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=6)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=7)


xlabel!(density_plot1, "Sample values",labelfontsize=10,subplot=1)
# plot!(density_plot1,[NaN], label="95% CI", color=:red,lw=lws,  linestyle=:dash, subplot=1)
# vline!(density_plot1,[NaN], lengend=true, linestyle=:dash, color=corbbf1std104, label="95% CI", subplot=1)


true_params=[1e7, 1e7, 1, 0.5, 0.1, 0.1, 40.5].^2
density_plot2 = density(Array(all_chains_bf[1][:,9:15].^2),  alpha = va, c=corbbf1, lenged=false ,xlabel="", ylabel=ll, layout=(7,1), size=(600,900))
for i in 2:100
    density!(Array(all_chains_bf[i][:,9:15].^2),  c=corbbf1,alpha = va, legend=false)
end
for (i, true_val) in enumerate(true_params)
    for ci95 in hcat(all_ci_bf...)'[:,i]
        vline!(ci95, linestyle=:dash, color=corbbf1,  alpha = va, label="95% CI",layout=(6,1), subplot=i)
    end
    #mean of all credible interval 95
    # vline!(mean(hcat(all_ci_bf...)'[:,i]), linestyle=:dash, color=corbbf1, label="95% CI",layout=(6,1), subplot=i)
    vline!(density_plot2, [true_val], lw=2, lc=:red, layout=(7,1), subplot=i)
end

title!(density_plot2, "Densities of CBM", subplot=1,titlefontsize=12)
xticks!(density_plot2, [2.5e15,15e15], subplot=1)
xticks!(density_plot2, [2.5e15,15e15], subplot=2)
xticks!(density_plot2, [0.01,0.02,0.03], subplot=6)
xticks!(density_plot2, [1000,2000,3000], subplot=7)
yticks!(density_plot2, [0,4e-16], subplot=1)
yticks!(density_plot2, [0,4e-16], subplot=2)
# yticks!(density_plot2, [1e-8,3e-8,6e-8,], subplot=2)
yticks!(density_plot2, [0,1.25,2.5], subplot=3)
yticks!(density_plot2, [0,5,10], subplot=4)
yticks!(density_plot2, [0,0.001,0.002], subplot=7)

# xticks!(density_plot2, [0.0089, 0.0099], subplot=3)
# xticks!(density_plot2, [0.4, 0.42, 0.44, 0.46], subplot=5)
# xticks!(density_plot2, [6.9e-9,  7.4e-9], subplot=6)
# yticks!(density_plot2, [0,1000,2500], subplot=3)
# yticks!(density_plot2, [0,20,40,60], subplot=5)
#
xlabel!(density_plot2, "Sample values", subplot=7,labelfontsize=10)

lws=3
# pltL1 = vline([NaN], label="95% Credible Interval", color=corbbf1std104,lw=lws,  linestyle=:dot)
# vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
# plot!([NaN], label="Ground truth value", color=:red, lw=lws)
#
# pltL2 = vline([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
# vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
# plot!([NaN], label="Ground truth value", color=:red, lw=lws)

plot1l1 = plot(density_plot1, size=(600, 800),xtickfontsize=10,ytickfontsize=10,yguidefontsize=16, legend=false)
plot2l2 = plot(density_plot2, size=(600, 800),xtickfontsize=10,ytickfontsize=10, yguidefontsize=16, legend=false)

allplots = plot(plot1l1,plot2l2, layout=(1,2))
plot!(left_margin=5mm,right_margin=5mm, bottom_margin=2mm,grid=false)
display(allplots)


# savefig(FULL_PATH*"/SBM/empirical_tests/results_analysis/estimated_sigmas_rq4.png")




##
# estimated noises SBM with variance and CBM with variance WITH HISTAGRAMS
#

all_ci_bbf1=[]
for i in 1:101
    civ=[credible_interval2(all_chains_bbf1[i][:,9])]
    append!(all_ci_bbf1,[civ])
end
all_ci_bf=[]
for i in 1:100
    civ=[credible_interval2(all_chains_bf[i][:,9].^2),
        credible_interval2(all_chains_bf[i][:,10].^2),
        credible_interval2(all_chains_bf[i][:,11].^2),
        credible_interval2(all_chains_bf[i][:,12].^2),
        credible_interval2(all_chains_bf[i][:,13].^2),
        credible_interval2(all_chains_bf[i][:,14].^2),
        credible_interval2(all_chains_bf[i][:,15].^2)]
    append!(all_ci_bf,[civ])
end
corbbf1=:purple2
corbbf1std104=:royalblue1


ll=[L"\sigma_{Xv}^2" L"\sigma_{Xt}^2" L"\sigma_{GLC}^2" L"\sigma_{GLN}^2" L"\sigma_{LAC}^2" L"\sigma_{AMM}^2" L"\sigma_{mAb}^2"]
true_params= [64083.3*0.1] #std value used to create the narrow surrogate model-> N(64083.3,64083.3*0.1)
#plot to answer RQ4
va=0.095
density_plot1 = stephist(Array(all_chains_bbf1[1][:,9:9]),  bins=:auto, alpha = va, c=corbbf1std104, label=L"\sigma^2" ,xlabel="", ylabel=L"\sigma_{\nu}^2", layout=(7,1), size=(600,900), legend=false)
for i in 2:100
    stephist!(Array(all_chains_bbf1[i][:,9:9]),  bins=:auto, c=corbbf1std104,alpha = va, label=L"\sigma^2", legend=false)
end
vline!(density_plot1, true_params, lw=2, lc=:red, layout=(7,1), subplot=1)
#
# for (i, true_val) in enumerate(true_params)
for ci95 in hcat(all_ci_bbf1...)'[:,1]
    #mean of all credible interval 95
    vline!(ci95, linestyle=:dash,  alpha = 0.5, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=1)
end
#     #mean of all credible interval 95
#     # vline!(mean(hcat(all_ci_bbf1...)'[:,i]), linestyle=:dash, color=corbbf1std104, label="95% CI",layout=(6,1), subplot=i)
#     vline!(density_plot1, [true_val], lw=2, lc=:red, layout=(7,1), subplot=i)
# end

# title!(density_plot1, L"SBM-BBF1~(4~\nu_{min}~values)", subplot=1,titlefontsize=12)
title!(density_plot1, "Histograms from SBM-BBF1(NSDS4)", subplot=1,titlefontsize=12)
xticks!(density_plot1, [1e3,25e3,5e4], subplot=1)
xaxis!(density_plot1, [1e3, 5e4],subplot=1)
# yticks!(density_plot1, [0,0.00005,0.0001], subplot=1)

# yticks!(density_plot1, [0,1e8,2.5e8], subplot=6) , legend=L"\sigma^2"

# density([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
# plot!(density_plot1,legend=L"\sigma^2",subplot=1)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=2, legend=false)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=3, legend=false)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=4, legend=false)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=5, legend=false)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=6, legend=false)
plot!(density_plot1,[NaN], xaxis=false, yaxis=false, xtick=false, ytick=false,ylabel="", lenged=false, grid=false,subplot=7, legend=false)


xlabel!(density_plot1, "Sample values",labelfontsize=10,subplot=1)
# plot!(density_plot1,[NaN], label="95% CI", color=:red,lw=lws,  linestyle=:dash, subplot=1)
# vline!(density_plot1,[NaN], lengend=true, linestyle=:dash, color=corbbf1std104, label="95% CI", subplot=1)


# true_params=[1e7, 1e7, 1, 0.5, 0.1, 0.1, 40.5].^2
true_params=[1e7, 1e7, 1, 0.5, 2.0, 0.1, 40.5].^2

density_plot2 = stephist(Array(all_chains_bf[1][:,9:15].^2),  bins=[:scott :sqrt :auto :auto :auto :sqrt :sqrt ], alpha = va, c=corbbf1, lenged=false ,xlabel="", ylabel=ll, layout=(7,1), size=(600,900))
for i in 2:100
    stephist!(Array(all_chains_bf[i][:,9:15].^2),  bins=[:scott :sqrt :auto :auto :auto :sqrt :sqrt ], c=corbbf1,alpha = va, legend=false)
end
for (i, true_val) in enumerate(true_params)
    for ci95 in hcat(all_ci_bf...)'[:,i]
        vline!(ci95, linestyle=:dash, color=corbbf1,  alpha = va, label="95% CI",layout=(6,1), subplot=i)
    end
    #mean of all credible interval 95
    # vline!(mean(hcat(all_ci_bf...)'[:,i]), linestyle=:dash, color=corbbf1, label="95% CI",layout=(6,1), subplot=i)
    vline!(density_plot2, [true_val], lw=2, lc=:red, layout=(7,1), subplot=i)
end

title!(density_plot2, "Histograms from CBM", subplot=1,titlefontsize=12)
xticks!(density_plot2, [2.5e15,15e15], subplot=1)
xticks!(density_plot2, [2.5e15,15e15], subplot=2)
xticks!(density_plot2, [0.01,0.02,0.03], subplot=6)
xticks!(density_plot2, [0.75,1.5,2.25], subplot=3)
xticks!(density_plot2, [1000,2000,3000], subplot=7)
yticks!(density_plot2, [0,250,500], subplot=1)
yticks!(density_plot2, [0,250,500], subplot=2)
# yticks!(density_plot2, [1e-8,3e-8,6e-8,], subplot=2)
# yticks!(density_plot2, [0,1.25,2.5], subplot=3)
# yticks!(density_plot2, [0,5,10], subplot=4)
# yticks!(density_plot2, [0,0.001,0.002], subplot=7)

# xticks!(density_plot2, [0.0089, 0.0099], subplot=3)
# xticks!(density_plot2, [0.4, 0.42, 0.44, 0.46], subplot=5)
# xticks!(density_plot2, [6.9e-9,  7.4e-9], subplot=6)
# yticks!(density_plot2, [0,1000,2500], subplot=3)
# yticks!(density_plot2, [0,20,40,60], subplot=5)
#
xlabel!(density_plot2, "Sample values", subplot=7,labelfontsize=10)

lws=3
pltL1 = vline([NaN], label="95% Credible Interval", color=corbbf1std104,lw=lws,  linestyle=:dot)
vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
plot!([NaN], label="Ground truth value", color=:red, lw=lws)

pltL2 = vline([NaN], label="95% Credible Interval", color=corbbf1, lw=lws, linestyle=:dot)
vline!(xaxis=false, yaxis=false, xtick=false, ytick=false, grid=false,  legend=:top,top_margin=-5mm)
plot!([NaN], label="Ground truth value", color=:red, lw=lws)

plot1l1 = plot(density_plot1,pltL1,layout=(grid(2,1, heights=[0.96, 0.04])), size=(600, 850),xtickfontsize=10,ytickfontsize=10,yguidefontsize=16)
plot2l2 = plot(density_plot2,pltL2, layout=(grid(2,1, heights=[0.96, 0.04])), size=(600, 850),xtickfontsize=10,ytickfontsize=10, yguidefontsize=16)


# plot1l1 = plot(density_plot1, size=(600, 850),xtickfontsize=10,ytickfontsize=10,yguidefontsize=16, legend=false)
# plot2l2 = plot(density_plot2, size=(600, 850),xtickfontsize=10,ytickfontsize=10, yguidefontsize=16, legend=false)

allplots = plot(plot1l1,plot2l2, layout=(1,2))
plot!(left_margin=5mm,right_margin=5mm, bottom_margin=2mm,grid=false)
display(allplots)


savefig(FULL_PATH*"/SBM/empirical_tests/results_analysis/Figure_1_suplementart_material.png")
