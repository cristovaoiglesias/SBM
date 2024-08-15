using CSV, DataFrames
using Printf
using LinearAlgebra, DifferentialEquations
# using Plots; gr();
using Plots,Noise
using Infiltrator,StatsPlots
using Dates,Metaheuristics
using Distributions
using Turing
# using  OptimizationOptimJL, OptimizationFlux
# using  ModelingToolkit , Optimization
# using  Optimization, OptimizationBBO
# using Symbolics

using Random
Random.seed!(1);





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
noise_dataset=add_gauss(transpose(sol),0.0)
noise_dataset[:,1]=add_gauss(transpose(sol)[:,1],10e7) #specific noise used for Xv
noise_dataset[:,2]=add_gauss(transpose(sol)[:,2],10e7)#specific noise
noise_dataset[:,3]=add_gauss(transpose(sol)[:,3],1)
noise_dataset[:,4]=add_gauss(transpose(sol)[:,4],.5)
noise_dataset[:,5]=add_gauss(transpose(sol)[:,5],2.)
noise_dataset[:,6]=add_gauss(transpose(sol)[:,6],.1)
noise_dataset[:,7]=add_gauss(transpose(sol)[:,7],40.5)#specific noise
noise_dataset=transpose(noise_dataset)

noise_dataset=[2.622709882387273e8 2.0194279199307412e8 1.6167862119212228e8 3.276807738059374e8 3.7935371642829883e8 1.9770802422624606e8 2.1191326223125517e8 2.4603471819068947e8 1.911783039082254e8 2.2429212546780646e8 3.2337844113259435e8 1.7912152440434566e8 3.0621404853213334e8 3.23901914684908e8 4.481615330406361e8 3.2453631613262945e8 3.995372389597404e8 5.231141802128235e8 5.8285037674442e8 4.6858072833689153e8 5.1962459494106376e8 4.861737138903069e8 6.244432826000214e8 4.4065169090418935e8 5.986214674397179e8 6.766616672004606e8 3.709646472477536e8 7.532846455931472e8 6.8181755413375e8 5.765508975766034e8 6.851650168811543e8 9.141156331501462e8 8.627252147288188e8 7.363368849504142e8 8.346649889467467e8 8.785722081890684e8 8.067671081280519e8 1.0180572442968919e9 9.205809509686451e8 8.603079776096922e8 9.795287820056833e8 1.0677966596343756e9 9.81462034397675e8 1.1920366655360231e9 1.1670020484540496e9 1.2323557234927135e9 1.2815469560125215e9 1.1991743823479967e9 1.249561070686404e9 1.3886995160436933e9 1.1796457689233913e9 1.2561466060974743e9 1.3020708597075052e9 1.3466064891768904e9 1.2290585881126304e9 1.2855600371265068e9 1.2552254262596998e9 1.5111157381179624e9 1.2650937759414754e9 1.3781036940813951e9 1.2314702667341425e9 1.1788340630956252e9 1.1345903346993806e9 1.2138756163002603e9 9.801906769518042e8 1.3761874346243305e9 1.154110435366774e9 1.03933307852024e9 9.152240606004863e8 1.1464449137679555e9 9.446681425145663e8 7.586967483857651e8 1.0147484064429291e9 1.02952408589797e9 1.0244610289589368e9 8.724187266558887e8 7.904463927813615e8 1.0360404494309562e9 9.302841755198325e8 7.935851289654316e8 8.442717983431426e8 7.930798705617493e8 7.967548342997178e8 7.184073667461457e8 7.817281196851547e8 6.819940907529664e8 6.443006212254937e8 6.030236857347547e8 5.828979407285898e8 8.241892464505432e8 7.506408983662498e8 5.50104427085231e8 6.658852375904856e8 6.551270026440921e8 5.2086278205869615e8 5.476629132298094e8 7.458493861149327e8 5.543397358500638e8 5.873736444261022e8 5.017270328448467e8 6.73445795764657e8 4.407982533762787e8 4.7703433059004384e8 5.1940485771500313e8; 2.1428819487901792e8 1.8372433151140982e8 3.737910616882963e8 1.782832293760267e8 3.6220385739934915e8 3.1419683289308566e8 4.542290842456165e8 4.436200788745836e8 1.5089871269999748e8 3.681445177139012e8 2.2789030391479635e8 3.1853466127063155e8 3.0179667278896546e8 5.726090159759759e8 3.993116783124364e8 3.766461074754633e8 3.878380216999912e8 4.263856677706928e8 3.593613590513307e8 5.082409061231533e8 4.420531307462955e8 5.56441291892079e8 5.967399776948133e8 5.643365297090917e8 7.745698721466902e8 6.875638873505797e8 6.673453151314206e8 5.203297018858541e8 6.872902752943093e8 7.62334370350743e8 6.856638652522278e8 7.248732220281342e8 9.281993797840326e8 9.202881948323021e8 9.537509522227796e8 1.0392754575223051e9 9.73746663989914e8 1.1068974018421142e9 9.477224677529842e8 1.1836422466716912e9 1.0637495400174644e9 1.1589314663276315e9 1.0652811356153684e9 1.211123403642006e9 1.3596512574505582e9 1.3745173906270013e9 1.3102394919110837e9 1.210679032615285e9 1.4004380684024396e9 1.5355367802789166e9 1.6514990018630588e9 1.6002682702361248e9 1.6382242082094622e9 1.639149346744396e9 1.6276609054027572e9 1.7716378154954658e9 1.6387636129476333e9 1.6439576736811554e9 1.6562457462924595e9 1.5951922843194017e9 1.68567675288444e9 1.737053251124392e9 1.6810230324578032e9 1.7515778964227867e9 1.5035273687464921e9 1.5962164244996865e9 1.4864163206352632e9 1.55182787388758e9 1.4002041148039637e9 1.5874862503886034e9 1.4794831837006693e9 1.3239143253534806e9 1.3371480502755296e9 1.3153480227161775e9 1.4116712573391113e9 1.0969704736637514e9 1.4144509782012947e9 1.2721350746430557e9 1.312892982181875e9 1.2568652429093337e9 1.179029147792027e9 1.1647482871651309e9 1.0405129589277313e9 1.2343576101991668e9 9.46948717786076e8 1.2148126268061314e9 1.135110019612623e9 1.2495066351735008e9 1.0080519938721552e9 1.0476459464002253e9 1.0969067439137506e9 8.346119560925379e8 9.361967502357726e8 9.156329028338491e8 8.899891893320222e8 8.797252198082141e8 7.579021771825961e8 8.719527065287284e8 8.477447708947272e8 9.021593049800278e8 8.117432305487416e8 7.626734895934641e8 8.335505499721094e8 8.32494286839952e8; 29.20277026467487 29.477294016173243 28.21136533085745 26.38632914178598 27.119271375526576 27.643571762919308 28.241335867321837 28.81220732920756 29.881534646210394 28.00663417619523 29.09436406801536 27.28395055962792 26.989455521506937 26.801287333502657 25.80994258736127 25.73434031065365 26.424107033727182 26.4276171980567 24.262557440866477 26.557581701658393 27.535032794135134 26.948786085337893 24.573786594819094 24.43816293357499 25.971886897460394 24.55654220222536 22.757632958110687 23.48747034401116 22.980383698601702 23.289630129849584 24.287185037368882 22.95032121616785 21.752457372057325 21.5483797310853 21.92840502857233 20.700518961894584 21.38259100727973 20.357710303002996 20.61089640280984 19.889216521914275 18.42601407070489 20.498728790224394 19.003672049955405 18.93727287664149 15.910204147824027 15.594844330584017 16.535239612677994 17.03695099070657 15.244473337577867 15.457005495717572 14.345705895403757 12.355697775024026 16.9526043049757 11.12461239401616 10.931534495648298 10.840439256819648 12.943903906054679 9.164862389117213 12.673889601610894 9.467850943858586 10.223999224178591 9.860384630790968 11.11755734569875 12.037396403159336 11.408918058572148 10.650726582006497 9.501068887636743 11.882558130355646 9.08667882107524 10.607850414988297 10.60656804779511 9.80815007631465 9.658696691028492 11.067867936679566 9.426114120856141 11.0306097781526 12.56103070578413 10.470290535105832 11.286138401919178 9.687108686580828 11.397376295471057 9.58644240716913 10.689087193401043 12.473133851394055 9.692609984933718 11.613905260674093 10.38565086969745 10.761442636777298 9.941933664863486 11.006346588521073 12.106533807816575 11.426514541740653 10.441163988975493 11.686075273208521 11.414215802176894 10.44952336755459 11.341378916818766 10.071013068505838 10.22603309896004 10.219143538835093 10.749916189537705 11.565999475969816 11.5468953372948 10.769645250371818; 4.572306405051611 4.75947737996425 4.8833740277223425 4.298826009235349 4.168923562109967 5.0105768158662265 4.723607326202752 4.017351744361896 4.1341806977497 3.296067929262998 4.565350452782476 4.560801956382956 4.757469841044863 3.312859625071665 3.656397131769241 3.8839312309124523 3.4239439325538217 3.6067502487534036 3.7243501836262562 3.480970305095409 3.148308623895112 1.7968631650502027 3.6377252783116782 2.623966042324539 2.493800256466182 2.843380737103265 3.1000462131062734 2.6666921346819397 3.124999014046541 2.7540399177203625 2.486298315705401 2.2970985885238213 1.6872621846665816 2.1476033468300084 1.9330540951433186 2.329442248267433 2.0433466610180346 3.084456070007492 1.7338431651620252 1.7502287982684288 1.5094462274525926 1.6767407724203367 1.595008660037029 0.5066397288323892 1.3125457512941952 1.3586230807976756 0.6040404653184277 0.8681928026152348 1.8165974169983388 0.8465842817877771 1.2965318855838621 1.2401053085682936 1.6187147497317833 -0.1828377160955612 0.03179530153618404 0.29152381232712443 -0.2112016581271375 0.42841074223788533 -1.1571118754972995 -1.0193445294214019 -0.3830920589875316 -0.40437536242871897 0.967150273776057 0.8278487263611076 0.44968293278926125 0.6665985648412674 0.3590778788464661 -0.39959156994829703 -0.10476743705722334 -0.47606465531997016 0.25104686633812323 0.4814313752907306 -0.4545439847879182 -0.6839477400118419 -0.4306411705162389 0.6337645870332279 0.5933484985513895 0.8452798447245613 0.8228013467913725 -0.2468527082764849 0.08616319325275383 0.09982930190722672 -0.8014753242119892 0.09946040287067172 0.4537957975342864 0.7396067326025452 -0.35394442945054305 -0.4485442727736141 -0.15698430574590605 0.10443778151450508 -0.5657949508423331 -0.3522196547364423 -0.0801957682753377 0.18726754438504276 0.9553925669402515 0.05168693374112474 0.1849555745059573 -0.9752061328360352 0.6698241896955822 -0.27333573484040424 0.48513725442025174 0.0824560785355206 0.21209655376798217 -0.2506040182871947; -2.8917636470149763 -1.9223439095389163 -1.7237203791743667 3.75052028510859 -3.586579107998009 -0.7283735034245234 1.365458751697759 -0.44963796623711727 1.257422585788416 -2.4627621905612562 5.238521846533116 -1.3777000783465487 -0.953565112209724 -0.0847259085155625 2.9741448451910344 5.53692854038775 4.105846602868125 6.67363445441803 -0.47952179235975567 4.3422745573663795 7.16447623850906 5.804800905096104 5.199323415238607 4.969217461440027 5.590596412794953 5.332359661650347 6.1600132951910584 7.50768359575427 5.252141745537713 10.227165364503584 9.440641615900073 9.11993736861247 7.88598492782977 7.334926659680082 10.953309835979624 9.872776678898582 8.604473464257378 14.938527704415186 14.909999288601703 12.136615963885303 12.886119154617285 18.048672113366344 15.477555080076941 15.394726654331626 18.939530386199095 12.411734155358097 18.158194030332865 18.889748163660744 15.968064122252489 20.040717661221926 19.89147109380299 19.500377834962755 19.511390378948185 23.186092278708273 19.460430803777903 27.323430646054838 25.504024870253275 23.772381110864853 28.018452646895113 20.52343836349276 26.03158227308219 25.73399690786045 22.10241428567713 23.134361871046142 29.582956602925808 23.731958558806497 23.684540931651675 24.462181571516474 25.846063114469466 25.093026805053864 25.67211720223557 22.824447962892062 26.150425804545034 25.894764340439792 25.990808976290968 22.664993438067487 23.870368476492345 24.854400236279762 25.73808788341794 23.09874399717675 27.039577304248926 26.248680484987066 22.009203964592444 25.29095932237809 19.478906595668175 22.840048083389583 24.27955385006397 26.164516821157704 25.48494831611489 26.249308201170436 25.742764240795317 26.85387259202552 22.524871429244868 25.129447635185652 24.10373230085818 23.692759649912688 22.9258702864297 24.012938758345424 22.37777252861951 27.05627178461125 24.716862940316187 25.503632416644116 25.77202710834998 21.932708805723387; 0.25722460829921456 0.33498915123775597 0.4027836203177324 0.4691651850307952 0.5834812804019149 0.5352570256946412 0.9076805681305348 0.6401930595469243 0.9498987698989476 0.7386022110451885 0.8303342346605279 0.8097940754358821 0.9458701910383305 1.2026589427983705 1.0874502859878177 1.1342087522218576 1.1597620794396717 1.1165663468999247 1.118865692890915 1.4454994651182342 1.3791861524791802 1.4821180702742431 1.4095088789259655 1.6531971511011794 1.5676780195252344 1.7298646687167065 1.598761495665919 1.7224438723727509 1.9406074573036909 1.8927043683847402 1.7940054960538532 1.8571211094237792 2.1047140060067755 2.187138856976316 2.1041818767860048 1.8746537507034295 2.270225802295759 2.222166555633762 2.25402418026911 2.3124344063230002 2.6185775200476473 2.245620975917343 2.470767650580562 2.643924277414102 2.5971704534211066 2.607076994799372 2.622685432547014 2.762551841918591 2.8309471125407675 2.7902985995790535 2.8021633392540903 2.936905074652936 3.082776405796304 3.117229834848234 2.8800445502235914 3.196759200835972 3.208996609829755 3.2782110262064235 3.2814174546036607 3.1136407406226407 3.180139267853858 3.229676255539479 3.3256403191175825 3.2092293455155376 3.3977156676946887 3.1958702843468445 3.2871102736896898 3.1576901160268025 3.098217217232399 3.2556652048852595 3.295489654924931 3.2053184474028145 3.1928533704638116 3.0240907347199566 3.3477573332792683 3.2685201589821076 3.2362429624387703 3.2274513302280927 3.0604177096091383 3.2425211722641705 3.1964792204773897 2.996515061470883 2.9666524158758367 3.199632001931098 3.4184323470615166 3.087803651630721 3.421553547704577 3.4414529230467 3.3520396364765923 3.3003204027231083 3.1476920765614405 3.1983434308882193 3.125823171112305 3.328222945285563 3.2401340072726525 3.1570169516437856 3.1572037094861387 3.309635215665876 3.2319714855840056 3.37930515949849 3.118941894847676 3.321040991493803 3.1575371475088234 3.3330491950734102; 21.27600314809022 45.44065989653222 102.81408109206126 121.7133571100751 126.60738048774137 68.16041318501269 97.87234028658567 146.0279712406297 190.34198622801213 104.24046942857959 168.82256924990108 157.96909257555157 116.35920751643208 154.994908625421 158.5806532803913 101.9529908737392 114.40676436200026 181.24629282210418 139.88715051909674 101.0716289574107 124.38854824214512 183.67274395874728 202.4722226440912 213.78403236699415 258.3575379915837 180.91754716665974 187.4684392775224 226.5729590336052 219.91884975946692 255.49259396512483 258.56305063929165 263.66704894001333 220.35554238433335 326.34732934994224 357.20728996534024 247.63075221299795 395.2876840059106 297.7434357940236 359.87272166102883 333.16207130833504 387.1325817800238 427.0006281479464 419.14156787331143 399.61535637383463 538.518154628835 542.1641695332054 466.73523348495075 436.5075624064259 477.74231739578073 509.09058735496467 573.0918167081153 525.5651684562987 505.29705512228816 567.4358336543971 559.8945200671025 679.5116190912001 605.2854957601689 738.5379329569663 728.9130534733492 718.5453474635716 749.1285647214121 851.1017555274574 793.2118259110408 794.5375955736874 804.2839507246968 854.0942742191095 845.8586088814296 914.5906329168736 871.1015383326073 826.0161095090772 943.341663233339 978.1398840860968 861.8688126741878 957.762719971393 965.6730122642296 939.4088520447183 965.5541888960452 1006.3263528327203 999.4600504074475 1024.7101411085685 1067.7381722244547 1036.0104787117173 1018.662764676286 1070.6698717202837 1044.7741516412955 1078.4949598460787 1079.351766620866 1158.0909039298983 1091.4238596884416 1159.262546606043 1125.5769221729424 1177.1385624803788 1172.3618838689665 1210.2376209328852 1169.483996937815 1185.88069195908 1271.7986487967657 1247.0859072981918 1192.5694958386684 1236.6657093917363 1209.892770785825 1287.125116623224 1298.5454055840344 1208.8423544510729]

plots=
scatter(sol.t,transpose(noise_dataset), size=(1000,800),color=:lightblue, lw=1.5, label = "Noise", ylabel=["[Xv]" "[Xd]"  "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,3))
plot!(sol.t,transpose(sol),color=[:red :red :red :red :red :red :red], lw=3.5, label = "True", ylabel=["[Xv]" "[Xd]"  "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,3))
display(plots)



function BBF1(newp)
    mu_max, Klysis,Kdgln, Ylacglc, Yammgln, lambda =newp
              # mu_max, Kglc, Kgln, KIlac, KIamm, mu_dmax, Kdamm, Klysis, Yxglc,   mglc,     Yxgln,  alpha1,  alpha2, Kdgln, Ylacglc, Yammgln, r1, r2 ,lambda = p
    newparms = [mu_max, 0.75, 0.075, 171.756, 28.484, 3e-2, 1.76, Klysis, 1.061e8, 4.853e-14, 5.57e8, 3.4e-13, 4,     Kdgln, Ylacglc, Yammgln, 0.1, 2, lambda ]
    newprob = remake(prob, p = newparms)
    # sol = solve(newprob, AutoTsit5(Rosenbrock23()),saveat = tgrid)
    sol = solve(newprob, saveat = tgrid)

    M=Diagonal([1e7, 1e7, 1, 0.5, 0.1, 0.1, 40.5].^2)
    rpe= noise_dataset.-sol
    v=0
    for i=1:size(rpe)[2]
        v=v+(rpe[:,i])'*inv(M)*(rpe[:,i])
    end
    return v#, sol
end

function BBF1dt(newp,dt)
    mu_max, Klysis,Kdgln, Ylacglc, Yammgln, lambda =newp
              # mu_max, Kglc, Kgln, KIlac, KIamm, mu_dmax, Kdamm, Klysis, Yxglc,   mglc,     Yxgln,  alpha1,  alpha2, Kdgln, Ylacglc, Yammgln, r1, r2 ,lambda = p
    newparms = [mu_max, 0.75, 0.075, 171.756, 28.484, 3e-2, 1.76, Klysis, 1.061e8, 4.853e-14, 5.57e8, 3.4e-13, 4,     Kdgln, Ylacglc, Yammgln, 0.1, 2, lambda ]
    newprob = remake(prob, p = newparms)
    # sol = solve(newprob, AutoTsit5(Rosenbrock23()),saveat = tgrid)
    sol = solve(newprob, saveat = tgrid)

    M=Diagonal([1e7, 1e7, 1, 0.5, 0.1, 0.1, 40.5].^2)
    rpe= dt.-sol
    v=0
    for i=1:size(rpe)[2]
        v=v+(rpe[:,i])'*inv(M)*(rpe[:,i])
    end
    return v#, sol
end

function BBF2(newp)
    mu_max, Klysis,Kdgln, Ylacglc, Yammgln, lambda =newp
              # mu_max, Kglc, Kgln, KIlac, KIamm, mu_dmax, Kdamm, Klysis, Yxglc,   mglc,     Yxgln,  alpha1,  alpha2, Kdgln, Ylacglc, Yammgln, r1, r2 ,lambda = p
    newparms = [mu_max, 0.75, 0.075, 171.756, 28.484, 3e-2, 1.76, Klysis, 1.061e8, 4.853e-14, 5.57e8, 3.4e-13, 4,     Kdgln, Ylacglc, Yammgln, 0.1, 2, lambda ]
    newprob = remake(prob, p = newparms)
    sol = solve(newprob, saveat = tgrid)

    M=Diagonal([1e7, 1e7, 1, 0.5, 0.1, 0.1, 40.5].^2)
    rpe= noise_dataset.-sol
    v=0
    for i=1:size(rpe)[2]
        v=v+(rpe[:,i])'*(rpe[:,i])
    end
    return v#, sol
end

function BBF2dt(newp,dt)
    mu_max, Klysis,Kdgln, Ylacglc, Yammgln, lambda =newp
              # mu_max, Kglc, Kgln, KIlac, KIamm, mu_dmax, Kdamm, Klysis, Yxglc,   mglc,     Yxgln,  alpha1,  alpha2, Kdgln, Ylacglc, Yammgln, r1, r2 ,lambda = p
    newparms = [mu_max, 0.75, 0.075, 171.756, 28.484, 3e-2, 1.76, Klysis, 1.061e8, 4.853e-14, 5.57e8, 3.4e-13, 4,     Kdgln, Ylacglc, Yammgln, 0.1, 2, lambda ]
    newprob = remake(prob, p = newparms)
    sol = solve(newprob, saveat = tgrid)

    M=Diagonal([1e7, 1e7, 1, 0.5, 0.1, 0.1, 40.5].^2)
    rpe= dt.-sol
    v=0
    for i=1:size(rpe)[2]
        v=v+(rpe[:,i])'*(rpe[:,i])
    end
    return v#, sol
end



true_params= [5.8e-2,0.05511,9.6e-3,1.399,4.27e-1,7.21e-9]
lbb = [0.01 ,0.01, 0.001 ,0.1 ,0.027  ,0.21e-9]
ubb = [0.1  ,0.1 , 0.96  ,2.0 ,1.427 ,100.21e-9]
bounds = boxconstraints(lb = lbb, ub =ubb )
#

@model function mcmc_model(bbf,Vmin_dist,dt)
    mu_max ~ Uniform(0.01 , 0.1 )
    Klysis ~ Uniform(0.01 , 0.1 )
    Kdgln ~ Uniform(0.001 , 0.96 )
    Ylacglc ~ Uniform(0.1 , 2.0)
    Yammgln ~ Uniform(0.027 , 1.427)
    lambda ~ Uniform(  0.21e-9, 100.21e-9)

    params_mcmc=[mu_max, Klysis, Kdgln, Ylacglc, Yammgln, lambda]

    v = bbf(params_mcmc,dt)
    ss ~ InverseGamma(1, 3)
    # ss=0.25
    for i=1:length(Vmin_dist)
        Vmin_dist[i] ~ truncated(Normal(v, ss),0,10e100)
    end
end











println("\n ParticleSwarm for BBF2: ")


estimated_params=[0.05867970703192592, 0.05568563260446636, 0.010692658487332777, 2.0, 0.2912666959299186, 5.85115563182354e-8]
# v_min=BBF2(minimizer(estimated_params))
v_min=BBF2dt(estimated_params,noise_dataset)
println("v_min from PS :",v_min)
Vmin_dist = rand(truncated(Normal(v_min,v_min*0.1),0,1e40), 104)
println("Vmin_dist :",Vmin_dist)
Vmin_dist =[1.7616296140151004e18, 1.8458702732304084e18, 1.6564569899084572e18, 1.5481372398038077e18, 1.9048021127443717e18, 1.3258769640767918e18, 1.7872579495326766e18, 1.815312949735022e18, 1.5692209051533988e18, 1.7156410907063752e18, 1.806002524263499e18, 1.624095312163172e18, 1.5168280970976266e18, 1.451664580900491e18, 1.7918587918357996e18, 1.611929182876118e18, 1.5441365229040486e18, 1.82908823979434e18, 1.4029668294227318e18, 1.5367146841971676e18, 1.9698214925952323e18, 2.2986865929930394e18, 1.8277099881507988e18, 1.9368573639249485e18, 1.9349935574457009e18, 1.8683403259283935e18, 1.4280433237743839e18, 1.6105588070678377e18, 1.6648266851650683e18, 1.6608873971962547e18, 1.4465247873748234e18, 2.1804103820995208e18, 1.7187148702410762e18, 1.4081727419375672e18, 1.523862963680167e18, 1.7783850602666742e18, 1.9018127269051796e18, 1.640450621501934e18, 2.0257527522400387e18, 1.6056786455864799e18, 2.0735500257802842e18, 1.9069819686977697e18, 1.3063189327547182e18, 2.0597225665368084e18, 1.682610390513185e18, 1.6008314783942415e18, 1.3086041098485023e18, 1.632051106892879e18, 1.4297126930493814e18, 1.307828073949032e18, 1.4838955077893814e18, 1.62558817170784e18, 1.578897370832188e18, 2.1099832116955013e18, 1.747553925247275e18, 1.5606106669723932e18, 1.7549769376012972e18, 1.4945226374135447e18, 1.718669524178507e18, 1.7862412327710019e18, 1.7132046066359127e18, 1.8320991232472924e18, 1.4780427255802386e18, 1.511524505789672e18, 1.7670888261353805e18, 2.0108365106331556e18, 1.956956530846924e18, 1.761974665738466e18, 1.7897510250299766e18, 1.927397147388358e18, 1.8586391468339932e18, 1.6224995453615386e18, 1.974060891140824e18, 1.7794623873207542e18, 2.0697291158777121e18, 1.6996754753997842e18, 1.7116919855926612e18, 1.9503438264212864e18, 1.6290660795149279e18, 1.8608085360166646e18, 1.5553188040812675e18, 1.446710112254058e18, 1.5148197259709993e18, 1.8211960955376074e18, 1.9578997483734136e18, 1.4927545252687135e18, 1.7342088273221245e18, 1.8640270091231724e18, 1.8920038717613763e18, 1.6505611226474888e18, 1.642717809703529e18, 1.6253296911020083e18, 1.737207769516646e18, 1.4825242595881006e18, 1.8925899150087892e18, 1.7201098934271437e18, 1.992957355502189e18, 1.8108736968304832e18, 2.1130116899585308e18, 1.9138718594590374e18,
1.347553925247275e18, 1.9606106669723932e18, 1.5549769376012972e18, 1.7945226374135447e18]

# # Create the density plot
# density_plot = density(Vmin_dist, title="Density Plot", xlabel="Values", ylabel="Density")
# density!(rand(truncated(Normal(v_min,v_min*0.5),0,1e40), 104))
# display(density_plot)



# for i=1:104
#     chain = Turing.sample(mcmc_model(BBF2dt,Vmin_dist[1:i],noise_dataset), NUTS(.65),3000; progress=true)
#     display(chain)
#
#     gr();
#     plot(chain)
#     nm="params_dist_bbf2_$(i)_"*Dates.format(Dates.now(), "d u yyyy H:m:s")
#     println(nm)
#     savefig(nm)
#     plot(chain)
#     println("time total: ",chain.info.stop_time-chain.info.start_time)
#
#
#     chain_df = DataFrame(chain)
#     full_path_nm="chain_bbf2_$(i)_time_$(chain.info.stop_time-chain.info.start_time).csv"
#     println(full_path_nm)
#     CSV.write(full_path_nm, chain_df)
#
# end

# for ni in [1,2,3,4,5,10,25,50,75,104]
for ni in [25]
    all_time = []

    for i=1:101
        chain = Turing.sample(mcmc_model(BBF2dt,Vmin_dist[1:ni],noise_dataset), NUTS(.65),3000; progress=true)
        display(chain)

        gr();
        plot(chain)
        nm="params_dist_bbf2_$(ni)_"*Dates.format(Dates.now(), "d u yyyy H:m:s")
        println(nm)
        savefig(nm)
        plot(chain)
        println("time total: ",chain.info.stop_time-chain.info.start_time)

        append!(all_time,chain.info.stop_time-chain.info.start_time)

        chain_df = DataFrame(chain)
        full_path_nm="chain_bbf2_$(ni)_time_$(chain.info.stop_time-chain.info.start_time).csv"
        println(full_path_nm)
        CSV.write(full_path_nm, chain_df)

    end
    mmean=mean(all_time[2:end])
    sstd=std(all_time[2:end])
    append!(all_time,mmean)
    append!(all_time,sstd)

    allTime = DataFrame(time = all_time)
    CSV.write("/home/bolic/cris/Umin/empirical_test/exp2/set100/allTime_bbf2.2_$(ni).txt", allTime)

end
