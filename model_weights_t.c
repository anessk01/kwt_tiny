// Author: Aness Al-Qawlaq 
// Date: 21/2/2024
// University College Dublin

const float pos_embedding_weights_layer[324] = {-0.408357173204422, -1.0396270751953125, -0.5786798596382141, -1.4749085903167725, 0.7605863213539124, 0.15105129778385162, 0.6621005535125732, 0.5931835174560547, -0.2648114264011383, 0.8868167996406555, 0.055214785039424896, -0.5800194144248962, 0.5502064228057861, 1.0428372621536255, -0.27401024103164673, 0.3896782696247101, -0.5400910973548889, -2.240353584289551, -1.0780692100524902, 0.7051706314086914, -0.02532043866813183, 1.1618006229400635, 0.2236810028553009, 0.3906514644622803, 0.7147258520126343, 1.2181538343429565, 0.1759892702102661, -0.5959199666976929, -0.10047002136707306, -0.4090096950531006, -0.30191367864608765, 0.5360742807388306, 0.23475652933120728, 0.3483673632144928, 0.46341049671173096, 0.9379703402519226, -1.123645305633545, -1.2735587358474731, 0.21045629680156708, -0.9817121624946594, -1.2075555324554443, -0.01794767752289772, -0.3061160445213318, 0.4313969314098358, 0.08942391723394394, 0.7727809548377991, 0.8016926646232605, -0.4115671217441559, 0.563783586025238, 0.24404093623161316, 0.08852078765630722, 0.35401254892349243, 0.08053579181432724, -0.5601940155029297, -0.25885868072509766, -0.3883476257324219, 1.620963454246521, -0.7461716532707214, -0.522662878036499, -0.5644575357437134, 0.03489670157432556, 1.1145278215408325, 0.523641049861908, -0.43642082810401917, 1.6112849712371826, 0.49257880449295044, -0.48957276344299316, -0.42919811606407166, -0.1583232432603836, 0.5137150287628174, 0.47041192650794983, 0.10831105709075928, -0.4634305238723755, 1.0680723190307617, 0.2606353461742401, -0.605402946472168, 0.0483979731798172, 0.9749128818511963, 0.3893735110759735, -0.22723911702632904, -0.5489680171012878, -1.7813405990600586, -0.5029217004776001, 0.32339248061180115, 0.6340854167938232, 0.6619278788566589, -0.1854357272386551, -0.6688137650489807, -0.7238579988479614, 0.22049406170845032, 0.5221136212348938, 0.9820280075073242, -0.6997390389442444, 0.6058529019355774, -0.4954635202884674, 0.8961218595504761, 0.02376168966293335, -0.8477148413658142, 0.49724212288856506, -1.054715871810913, -0.3756522238254547, -0.0867295116186142, -0.2680398225784302, 0.4941351115703583, -0.5644941926002502, 0.35062313079833984, 0.32834815979003906, 0.3812619149684906, -0.7211928367614746, 0.42879384756088257, -0.9528868198394775, -1.1652324199676514, 0.4127001166343689, 0.9005457758903503, 0.4157801866531372, -0.9257736802101135, -0.19108638167381287, -0.04468793049454689, 0.31303519010543823, 0.6525675654411316, -0.3840969204902649, -0.21078114211559296, -1.1065417528152466, -0.49243101477622986, -0.9353722929954529, 0.9692140817642212, 0.24765628576278687, 0.3504146635532379, -2.078303337097168, -0.6946307420730591, 0.41246655583381653, 0.20342613756656647, 0.5326635241508484, 3.006143569946289, -0.1069316565990448, 0.19363559782505035, 1.876890778541565, 1.2887508869171143, -0.21483412384986877, -1.1705609560012817, 1.4592009782791138, -0.9961709380149841, -1.8005836009979248, -1.592983365058899, -0.1325681060552597, 0.008276325650513172, 0.4196087419986725, -1.2983423471450806, 0.4861658215522766, 1.2096015214920044, 0.8668822050094604, 0.8931152820587158, 0.3611180782318115, -1.0866379737854004, 0.599082887172699, -1.1530343294143677, 0.7403700947761536, 0.007438197266310453, 0.48627904057502747, -0.7654462456703186, 0.9908684492111206, -0.08527359366416931, 1.6695163249969482, 0.039870209991931915, 0.25256508588790894, 0.6838501691818237, 0.3164103031158447, 1.0426384210586548, 0.9691009521484375, 2.038069486618042, 0.18186232447624207, 0.19690582156181335, -0.22402982413768768, -0.41757938265800476, -0.31988632678985596, -1.0351117849349976, -0.8242083787918091, -0.07544497400522232, -1.1225546598434448, -1.38517427444458, 0.3070690631866455, 0.632686197757721, 0.9737542271614075, 0.6831574440002441, 1.6721762418746948, -0.7495317459106445, -0.40258246660232544, 1.2398691177368164, 1.1153130531311035, 3.160282611846924, 0.8931645154953003, -1.2382136583328247, -0.37600141763687134, 0.6385849118232727, 0.5254936218261719, 1.1329983472824097, -0.5492216348648071, 0.17684051394462585, 0.17649580538272858, -1.7424896955490112, -1.458335280418396, 0.5534220337867737, -0.04717208445072174, -1.1561169624328613, 1.3241264820098877, -0.9017288088798523, 0.5827474594116211, 0.9005289673805237, -1.6664801836013794, 0.3868526220321655, -0.39851900935173035, -0.07419771701097488, -0.05013839155435562, 0.7094660997390747, -0.81455397605896, 1.3277232646942139, -0.22686731815338135, 0.95383620262146, -0.010088643059134483, -0.7529730796813965, -0.6946921944618225, 0.23106065392494202, 0.5743294358253479, 1.3255754709243774, -0.4654897153377533, 0.7984992265701294, 0.03771767020225525, 1.079055666923523, 1.8350592851638794, 0.9625547528266907, -0.3550982177257538, 0.27094218134880066, -0.33765485882759094, 0.7943867444992065, -0.37786298990249634, 0.2670558989048004, 0.1896856129169464, 0.08784689754247665, 0.29651594161987305, -0.630709707736969, 0.2005005031824112, -0.011128120124340057, 0.26787686347961426, 0.9369437098503113, -0.7856206893920898, -0.05960883945226669, 1.45266592502594, -0.8287565112113953, -0.11471742391586304, -1.613175630569458, 0.5344446897506714, -0.6453403830528259, -0.18407978117465973, -0.2899869680404663, 0.9455806612968445, 1.0411796569824219, 0.11927322298288345, -0.35571858286857605, -0.20105139911174774, 0.3885200321674347, 0.24626439809799194, -1.211937427520752, -0.5115432143211365, 0.4544197916984558, 0.015144092962145805, 0.5359302163124084, -1.0420750379562378, -0.0045142085291445255, -1.0302362442016602, 0.4315532445907593, -0.8797934055328369, 0.012707680463790894, -0.2916554808616638, 0.9524188041687012, 0.22255948185920715, 0.1916622817516327, -0.5027053356170654, -0.3929632902145386, -0.29784172773361206, -1.14169180393219, 0.11352609843015671, -0.8369033336639404, -0.5057465434074402, -1.2364901304244995, -0.1094389259815216, 0.506219744682312, 0.18256323039531708, -0.5372693538665771, 1.0697740316390991, 0.22030438482761383, 0.07792747020721436, 0.28661638498306274, 0.25148630142211914, 0.040043264627456665, -0.9919975996017456, -0.5531064867973328, 0.40831729769706726, -0.35756102204322815, -1.694887638092041, -0.7217891216278076, -0.5018003582954407, 0.4409268796443939, -1.491758942604065, -1.7422749996185303, -0.10719244927167892, -1.1455472707748413, 0.11021732538938522, -1.2459442615509033, -0.4598548710346222, 0.3054139316082001, -1.2123024463653564, -1.7334505319595337, -0.31443262100219727, -0.582929253578186, -0.34960469603538513, 0.15996244549751282, 0.6094019412994385, 0.37320876121520996, 2.465527296066284, -1.0775967836380005, 0.4830342233181, -0.04870850220322609, 0.2308449149131775, 1.0011985301971436};

const float cls_token_weights_layer[12] = {-0.534445583820343, -0.10634486377239227, -0.41725829243659973, 0.045081816613674164, 1.0854920148849487, -0.39627575874328613, 0.448945015668869, 0.7498204112052917, 0.8018949031829834, 0.5749173760414124, -0.27422043681144714, 0.29418081045150757};

const float to_patch_embedding_1_weight_weights_layer[192] = {0.055817462503910065, 0.0736561268568039, -0.06473517417907715, -0.07016737014055252, 0.05839524418115616, -0.03990945592522621, 0.06101282685995102, -0.030091138556599617, -0.07356642186641693, -0.09622882306575775, -0.03117009624838829, -0.059137530624866486, 0.15619195997714996, 0.10146025568246841, -0.1736670732498169, -0.13977859914302826, 0.11693799495697021, 0.06157083809375763, 0.12992531061172485, 0.02759844996035099, -0.2103712558746338, -0.08060528337955475, -0.10208354145288467, -0.14179466664791107, -0.1895013153553009, -0.12790144979953766, -0.1418730765581131, 0.2287517786026001, 0.011044970713555813, 0.1835450679063797, -0.21465961635112762, 0.22825735807418823, -0.03475864976644516, 0.0827602744102478, -0.12848663330078125, 0.1319999396800995, -0.1175932064652443, 0.11186610162258148, 0.06813429296016693, 0.04272418096661568, -0.03705746307969093, -0.12497109174728394, 0.03137368708848953, -0.27717068791389465, 0.2022128403186798, -0.14610151946544647, -0.10981755703687668, -0.0987982451915741, -0.15042683482170105, 0.07198016345500946, -0.12682433426380157, 0.1726139634847641, 0.004334159195423126, 0.09871231019496918, 0.04273057356476784, 0.06838519126176834, -0.07528340071439743, -0.0997144877910614, -0.09810011088848114, 0.05382125452160835, -0.09063500910997391, 0.24844519793987274, 0.4395763874053955, -0.051201388239860535, 0.08974955976009369, 0.012190969660878181, 0.278883159160614, -0.08527576178312302, 0.2107250839471817, -0.2015528678894043, 0.07891497761011124, 0.05323655530810356, 0.1359567493200302, 0.09973426163196564, 0.27452224493026733, -0.011917483992874622, -0.206760436296463, -0.017915261909365654, 0.06433748453855515, -0.10680520534515381, 0.2506352961063385, -0.23152485489845276, 0.00941268540918827, 0.18343724310398102, -0.05950571224093437, 0.11950080096721649, 0.16772882640361786, -0.04371766373515129, -0.17364253103733063, -0.13625279068946838, -0.11213050782680511, 0.08908132463693619, 0.1333177387714386, -0.1771683394908905, 0.11161768436431885, 0.22408509254455566, -0.084286630153656, -0.19122834503650665, -0.07003966718912125, 0.12429223209619522, 0.02793453261256218, 0.005689084064215422, -0.2891668975353241, 0.1546863615512848, -0.051721006631851196, 0.07670195400714874, -0.1284302920103073, 0.1580267697572708, 0.09970193356275558, -0.07469446212053299, -0.2589067816734314, -0.10644994676113129, 0.1985466480255127, -0.17549526691436768, -0.06657625734806061, 0.07575377076864243, -0.1644909679889679, 0.1647339165210724, 0.004230756778270006, 0.0645887479186058, 0.08454735577106476, 0.08082377910614014, -0.05473588407039642, -0.06748445332050323, 0.11965654045343399, 0.023477081209421158, -0.2670830190181732, -0.061077315360307693, -0.06482009589672089, 0.13768243789672852, 0.2179737091064453, 0.009238794445991516, -0.01286502368748188, -0.07279547303915024, -0.021142931655049324, -0.21226046979427338, 0.12422086298465729, 0.07696797698736191, -0.05373578146100044, -0.04494796693325043, 0.2663858234882355, -0.22862228751182556, 0.045924026519060135, 0.010940024629235268, -0.18332016468048096, -0.05934945493936539, -0.01797572150826454, 0.06287693977355957, -0.036121584475040436, 0.11938696354627609, -0.020973626524209976, -0.21777494251728058, 0.2718120515346527, 0.1053474172949791, 0.2394331693649292, -0.06435476243495941, 0.060018621385097504, 0.015680020675063133, -0.039794351905584335, -0.044296566396951675, -0.12522371113300323, 0.11777622252702713, 0.12266916781663895, -0.22081294655799866, 0.09542886167764664, -0.1489873081445694, 0.12932322919368744, 0.03805437311530113, -0.0544988289475441, -0.13392947614192963, 0.021196449175477028, -0.006640493404120207, -0.004340081941336393, -0.1350962072610855, -0.18967527151107788, 0.02898983471095562, 0.08747399598360062, -0.002366832923144102, -0.21297959983348846, 0.14386047422885895, -0.05441651493310928, 0.13957326114177704, 0.07599887996912003, 0.1497897356748581, -0.17088374495506287, -0.14608854055404663, 0.0017135787056759, -0.03801945596933365, -0.013267436996102333, -0.09914731979370117, 0.12922219932079315, 0.16502344608306885};

const float to_patch_embedding_1_bias_weights_layer[12] = {-0.14264382421970367, 0.01720530353486538, -0.0886920690536499, -0.145677387714386, -0.032622288912534714, 0.05017067492008209, -0.06418385356664658, -0.038601696491241455, 0.06823941320180893, 0.11116737127304077, -0.11588703095912933, -0.16873012483119965};

const float transformer_layers_0_0_norm_weight_weights_postnorm[12] = {0.8295177221298218, 0.877277672290802, 0.985458254814148, 0.7601400017738342, 0.880893886089325, 0.9616325497627258, 0.896101713180542, 0.8317927122116089, 0.7540909051895142, 0.8692439794540405, 0.8072339296340942, 0.830353856086731};

const float transformer_layers_0_0_norm_bias_weights_postnorm[12] = {0.0038748460356146097, 0.018089208751916885, -0.031794652342796326, 0.07112737745046616, -0.04863852635025978, 0.03249741718173027, -0.0206208024173975, -0.012661369517445564, 0.004742912482470274, -0.006213100627064705, -0.006570249330252409, 0.015844643115997314};

const float transformer_layers_0_0_fn_to_qkv_weight_weights_attention[288] = {0.12448995560407639, 0.030906736850738525, -0.19798080623149872, -0.14313946664333344, 0.1157815009355545, -0.1691514551639557, -0.1293044537305832, 0.11620615422725677, -0.03323286771774292, -0.001747697126120329, 0.028784310445189476, -0.11737768352031708, -0.24958071112632751, -0.1532755047082901, -0.09741204231977463, 0.07563034445047379, -0.035748835653066635, 0.07217151671648026, -0.15907198190689087, -0.12344352155923843, 0.11102882772684097, 0.19686159491539001, -0.027622265741229057, 0.00847986713051796, 0.12745681405067444, -0.11996541917324066, 0.15653900802135468, -0.08396752923727036, -0.017611194401979446, 0.016492098569869995, 0.03454705327749252, -0.1485462784767151, 0.21336866915225983, -0.01158137433230877, 0.034984588623046875, 0.1127859428524971, -0.20488032698631287, 0.05032602325081825, -0.1726807802915573, -0.014502232894301414, 0.06788194924592972, 0.18371863663196564, -0.07105930149555206, -0.13012777268886566, -0.05515068769454956, -0.0065055228769779205, -0.18393424153327942, -0.10235664993524551, 0.1984022855758667, -0.1360987424850464, 0.05559620261192322, 0.11629460006952286, 0.11576560884714127, 0.12727563083171844, 0.08797445148229599, -0.18514949083328247, -0.025483321398496628, 0.13260342180728912, 0.04599380120635033, -0.16213396191596985, 0.23000967502593994, 0.24668464064598083, 0.036419667303562164, -0.17304615676403046, -0.19025109708309174, 0.14490818977355957, -0.0802832618355751, -0.06079643592238426, -0.1854357123374939, 0.12498950958251953, -0.19499653577804565, 0.04622910916805267, 0.12644819915294647, 0.1158340647816658, 0.05241718515753746, -0.09410903602838516, -0.009627193212509155, 0.06755919009447098, 0.02011674828827381, -0.10285145044326782, 0.19636304676532745, -0.07884770631790161, -0.06437627971172333, 0.046034205704927444, -0.027725063264369965, -0.15093128383159637, -0.10290860384702682, -0.0876598134636879, 0.23811876773834229, -0.10066302865743637, -0.21469120681285858, -0.058501631021499634, -0.12352173030376434, -0.05908005312085152, 0.10486768931150436, 0.1913655698299408, -0.06852452456951141, -0.06875460594892502, -0.16199833154678345, -0.09943555295467377, -0.060681018978357315, 0.04996179789304733, -0.11662323772907257, 0.05556196719408035, -0.03572676330804825, 0.17811322212219238, 0.15202027559280396, -0.10569221526384354, 0.03661086782813072, 0.1409810483455658, -0.1508646309375763, 0.16888689994812012, 0.08438127487897873, 0.2134324163198471, 0.007240945938974619, 0.11235206574201584, 0.06025402247905731, -0.037743743509054184, 0.08221009373664856, 0.015378469601273537, -0.13808573782444, -0.04161843657493591, 0.173483207821846, -0.02998192608356476, 0.18239948153495789, 0.048449981957674026, 0.21382057666778564, 0.1929844617843628, 0.0753568708896637, -0.0695677399635315, 0.18720664083957672, -0.11628278344869614, -0.021677827462553978, -0.16820979118347168, -0.046944886445999146, 0.21729892492294312, -0.13299904763698578, -0.16225197911262512, 0.05887696146965027, -0.126434788107872, -0.14985065162181854, -0.10049518197774887, -0.10052413493394852, 0.0628575012087822, -0.021938065066933632, -0.02033347450196743, 0.23571579158306122, 0.006403164938092232, 0.07849745452404022, 0.07512100785970688, -0.14612731337547302, -0.07264294475317001, -0.20403631031513214, -0.024622038006782532, -0.03420194983482361, 0.06718724220991135, 0.15116210281848907, -0.0268417801707983, 0.15123803913593292, 0.0812586173415184, -0.06466232240200043, 0.23217666149139404, 0.05892176181077957, 0.16038456559181213, -0.034996263682842255, -0.00650057103484869, -0.14985090494155884, -0.019307255744934082, 0.19195739924907684, 0.001092186663299799, 0.21007144451141357, -0.13241520524024963, -0.19744156301021576, -0.06105771288275719, 0.16297182440757751, 0.1107044368982315, -0.07881001383066177, 0.031257666647434235, -0.1817643791437149, 0.09854558855295181, 0.14527902007102966, 0.1455611288547516, -0.07006797939538956, 0.047425881028175354, 0.17707867920398712, -0.19706420600414276, 0.0684533640742302, -0.13791866600513458, 0.11198878288269043, -0.0910911038517952, 0.06969437748193741, 0.17910462617874146, -0.08232247829437256, 0.11411678791046143, -0.1498476266860962, 0.1924189031124115, -0.0038132257759571075, -0.10374477505683899, -0.17613686621189117, -0.03989580646157265, 0.0068128351122140884, 0.044582657516002655, 0.11119244247674942, -0.030138224363327026, 0.23528549075126648, -0.008196808397769928, 0.11196894943714142, -0.13423217833042145, -0.1863163709640503, -0.08323731273412704, 0.16965632140636444, 0.013293237425386906, -0.21437954902648926, 0.1830601841211319, -0.10432939976453781, -0.16351301968097687, -0.1516498476266861, 0.08035305142402649, -0.08302760124206543, -0.05486912652850151, -0.1624249815940857, -0.21688780188560486, -0.062393080443143845, -0.005504056345671415, 0.08479861170053482, 0.13774801790714264, -0.11508551985025406, 0.11696097254753113, -0.14442504942417145, 0.05402081832289696, 0.22384479641914368, 0.08660639822483063, -0.04918554797768593, 0.1184467077255249, 0.0832880362868309, 0.19347992539405823, 0.17541997134685516, -0.03533671423792839, 0.20135879516601562, -0.09864199161529541, 0.04418996348977089, 0.06363299489021301, -0.13091526925563812, 0.012676564045250416, 0.19013771414756775, -0.17759522795677185, 0.1321513056755066, 0.11782950162887573, 0.08275087177753448, -0.043943047523498535, -0.010290191508829594, -0.009781453758478165, -0.03986988589167595, -0.13302049040794373, 0.10164561867713928, -0.05919322371482849, 0.06896339356899261, -0.03666742146015167, -0.17966832220554352, 0.21676012873649597, 0.03459441661834717, 0.16927391290664673, -0.009424258023500443, 0.027662018314003944, 0.21590010821819305, -0.13436049222946167, 0.18238070607185364, 0.1835840493440628, -0.06512206047773361, 0.09421046078205109, -0.08075971901416779, -0.18852408230304718, 0.19141723215579987, -0.20158356428146362, 0.08941220492124557, -0.11747083812952042, -0.10901974141597748, 0.18061509728431702, -0.06794829666614532, -0.21710610389709473, -0.15198436379432678, 0.007019019685685635, -0.10145410150289536, -0.13705064356327057, 0.1229785606265068, -0.054681066423654556, 0.1380600482225418, -0.15171603858470917};

const float transformer_layers_0_0_fn_to_out_0_weight_weights_attention[96] = {-0.022033043205738068, 0.20246846973896027, -0.16855493187904358, -0.057000525295734406, 0.08965391665697098, -0.25783273577690125, 0.19090987741947174, 0.04498108848929405, 0.10490575432777405, 0.17384997010231018, 0.17941699922084808, -0.11392464488744736, -0.11988852173089981, -0.15627089142799377, 0.007029606960713863, -0.2154129296541214, 0.14057578146457672, -0.038734886795282364, -0.06653840839862823, 0.0997684895992279, -0.20448791980743408, 0.1401689499616623, -0.22633005678653717, -0.034115519374608994, 0.1574401706457138, -0.029113570228219032, 0.004043810069561005, -0.18772131204605103, -0.11765524744987488, 0.08736863732337952, -0.2208610326051712, 0.2368888407945633, -0.26005131006240845, 0.31572505831718445, 0.19755633175373077, -0.2706785798072815, -0.23322449624538422, -0.09977247565984726, -0.012326660566031933, 0.26428619027137756, 0.18272621929645538, 0.08715186268091202, -0.09352558851242065, 0.09300793707370758, 0.1935654729604721, -0.21530352532863617, -0.24837180972099304, -0.2744090259075165, 0.0761224702000618, -0.08353741466999054, 0.20404325425624847, 0.1899406760931015, 0.11011222749948502, 0.3220725357532501, 0.0005002230755053461, -0.28833189606666565, 0.1645776331424713, -0.2930835485458374, 0.11647406220436096, 0.011678215116262436, 0.37601208686828613, 0.10129286348819733, 0.1944465935230255, 0.1843307912349701, 0.04184534400701523, 0.23667271435260773, -0.22860212624073029, -0.1502610445022583, -0.10249613970518112, 0.1555820107460022, -0.25667449831962585, -0.003098608460277319, -0.06329485028982162, 0.06437719613313675, -0.2200084626674652, 0.13853086531162262, 0.015401468612253666, 0.38149046897888184, 0.24501144886016846, 0.012793323956429958, 0.017478959634900093, 0.20645025372505188, -0.03994055464863777, -0.2932239770889282, -0.04830879345536232, -0.06627696752548218, 0.021907879039645195, -0.18410606682300568, 0.10147793591022491, 0.18334966897964478, 0.09356919676065445, 0.026530204340815544, -0.22290125489234924, 0.10292869061231613, 0.06108678877353668, -0.0698685348033905};

const float transformer_layers_0_0_fn_to_out_0_bias_weights_attention[12] = {-0.22416427731513977, -0.2847926616668701, 0.2309076488018036, 0.1413222700357437, 0.09436527639627457, -0.23215219378471375, 0.053640495985746384, 0.06799942255020142, -0.1897084265947342, 0.18061499297618866, 0.22181540727615356, -0.2631455063819885};

const float transformer_layers_0_1_norm_weight_weights_postnorm[12] = {0.7674340009689331, 0.8537480235099792, 0.8364336490631104, 0.8786004781723022, 0.7895826697349548, 0.7778599262237549, 0.8202974200248718, 0.7537766695022583, 0.7774427533149719, 0.9524834752082825, 0.7640371918678284, 0.7716051936149597};

const float transformer_layers_0_1_norm_bias_weights_postnorm[12] = {-0.007470732554793358, -0.0016177237266674638, -0.006304894108325243, 0.09695030003786087, -0.015543735586106777, 0.02617019973695278, -0.011877968907356262, 0.004275949206203222, 0.004900997970253229, 0.021561376750469208, 0.002152435015887022, -0.0005658250884152949};

const float transformer_layers_0_1_fn_net_0_weight_weights_feedforward[288] = {-0.20744876563549042, 0.014670833945274353, -0.07064317166805267, 0.00973560195416212, 0.1292145848274231, 0.0043509420938789845, -0.17636629939079285, -0.21959981322288513, -0.03657814487814903, 0.12001527100801468, 0.0673006922006607, 0.06812646239995956, 0.10215216130018234, 0.0004044264496769756, 0.013509843498468399, 0.12330947071313858, -0.06814221292734146, -0.11862746626138687, -0.035836465656757355, 0.12940160930156708, -0.14325934648513794, 0.11717823892831802, 0.08320575952529907, -0.16570843756198883, 0.03689813241362572, 0.1542784720659256, 0.13484179973602295, -0.1582798808813095, -0.12349803000688553, -0.1731567084789276, -0.053568970412015915, 0.11542492359876633, 0.26513251662254333, -0.13558252155780792, 0.14815101027488708, -0.15847398340702057, 0.05576345697045326, -0.1490350067615509, 0.19885443150997162, -0.19188635051250458, 0.23854531347751617, -0.15615852177143097, -0.08346476405858994, 0.21164964139461517, -0.0910235196352005, 0.10122168064117432, -0.16890573501586914, 0.1393725425004959, 0.21892113983631134, -0.3883042335510254, 0.05153347924351692, 0.037308644503355026, -0.00860852561891079, -0.15171116590499878, -0.27737683057785034, -0.03509785234928131, 0.13039274513721466, 0.2004980891942978, 0.1294662356376648, 0.05528217926621437, -0.12280304729938507, -0.1721171736717224, -0.12729071080684662, 0.12848690152168274, -0.0205792635679245, -0.011650875210762024, 0.18853136897087097, -0.009888799861073494, -0.13593855500221252, 0.1099993959069252, 0.1795324981212616, 0.1056903749704361, -0.1143665462732315, 0.06965361535549164, 0.15447162091732025, -0.1752171665430069, 0.13078266382217407, 0.10336972028017044, 0.09446383267641068, -0.07549374550580978, 0.24341534078121185, -0.2125784456729889, -0.036141932010650635, 0.035724788904190063, -0.17319242656230927, 0.043368883430957794, -0.13538405299186707, -0.17034229636192322, 0.11031538993120193, 0.17389832437038422, -0.20800191164016724, -0.10734519362449646, 0.2020769715309143, 0.010793184861540794, 0.18054918944835663, -0.18823321163654327, 0.05440341681241989, -0.021130776032805443, 0.04030391573905945, -0.07108230888843536, 0.17095516622066498, -0.07560335844755173, -0.1031780019402504, 0.15376023948192596, -0.16169795393943787, -0.2771937847137451, -0.003068925580009818, -0.07016712427139282, -0.11239062249660492, -0.23731854557991028, 0.009951766580343246, 0.08887484669685364, -0.27689650654792786, -0.1278870701789856, -0.16285084187984467, 0.11680229008197784, -0.02893347479403019, 0.19722910225391388, -0.08164527267217636, -0.13665494322776794, 0.09792550653219223, -0.0503581203520298, -0.10593961179256439, 0.02796834148466587, -0.007872234098613262, 0.1456640213727951, 0.3588830232620239, 0.22760792076587677, -0.09047839790582657, -0.10489217191934586, -0.2436806857585907, -0.018222006037831306, 0.10913312435150146, 0.03682180866599083, 0.13410548865795135, -0.2869146168231964, -0.060720983892679214, -0.26071876287460327, -0.23540866374969482, -0.00727382767945528, -0.1529378592967987, -0.006797795183956623, -0.17023292183876038, -0.21230553090572357, 0.0751437097787857, 0.1447993963956833, 0.16260460019111633, -0.11819086223840714, -0.1266764998435974, 0.19259481132030487, -0.06607545167207718, -0.24845673143863678, -0.06142624840140343, 0.1749277412891388, -0.04316306859254837, 0.16807322204113007, -0.232180655002594, -0.16105805337429047, 0.18922989070415497, 0.1398332566022873, -0.1770252287387848, -0.058839865028858185, -0.017761602997779846, 0.06816089153289795, -0.08080749213695526, -0.15095308423042297, 0.1493980586528778, -0.18675678968429565, 0.03651662543416023, -0.04983855411410332, 0.001323541859164834, -0.202296182513237, 0.030267994850873947, -0.010812012478709221, -0.0855756402015686, 0.008099468424916267, -0.2982599139213562, 0.009030317887663841, 0.14598537981510162, 0.11227388679981232, -0.2599945664405823, -0.17520132660865784, -0.17815455794334412, -0.2463490515947342, 0.15308091044425964, 0.020409133285284042, 0.1564817726612091, -0.07665976136922836, -0.12739752233028412, -0.004472739994525909, -0.16114726662635803, -0.1329229176044464, 0.1259695589542389, 0.22464029490947723, -0.07409925758838654, -0.11793538928031921, 0.098931685090065, 0.015108159743249416, 0.2516997754573822, 0.26931536197662354, 0.02878725156188011, 0.03924649953842163, -0.2007826566696167, -0.2064439058303833, 0.05703342705965042, 0.12275679409503937, 0.07927203923463821, -0.37193775177001953, 0.06701745837926865, -0.026301395148038864, 0.13531914353370667, 0.17858180403709412, 0.0694899633526802, -0.07005560398101807, -0.03178895637392998, 0.1634964793920517, 0.1007649227976799, -0.004303671419620514, -0.0865519791841507, -0.08669029921293259, -0.16254238784313202, -0.08347887545824051, 0.08395527303218842, 0.08742746710777283, 0.08045752346515656, -0.16703662276268005, 0.147422194480896, 0.020219212397933006, -0.18718110024929047, -0.19295251369476318, 0.15091288089752197, -0.02289046347141266, -0.17835864424705505, 0.0031422716565430164, -0.1916797012090683, 0.1781310737133026, -0.2072104513645172, 0.097307950258255, 0.06797000020742416, -0.05614432692527771, -0.08318661153316498, -0.10503708571195602, 0.016670912504196167, 0.021379975602030754, -0.12432513386011124, -0.04400842264294624, 0.09502950310707092, 0.20983345806598663, -0.05525502562522888, 0.08749926835298538, -0.06573233008384705, -0.22641083598136902, -0.20241938531398773, -0.17161434888839722, -0.20533224940299988, 0.15908996760845184, -0.012407172471284866, -0.0013667532475665212, 0.016115738078951836, 0.17081153392791748, -0.1880481243133545, -0.4692692458629608, 0.2057463824748993, 0.12154129892587662, 0.01769527606666088, -0.13599082827568054, -0.04231945425271988, 0.06885986775159836, 0.10569222271442413, 0.06279543787240982, 0.08635657280683517, -0.1001407578587532, 0.05296442657709122, -0.007894379086792469, -0.06548893451690674, -0.14142462611198425, -0.06301290541887283, 0.15695664286613464, -0.11995143443346024, -0.16253086924552917, 0.03429541736841202, -0.1694098860025406, 0.1237778440117836, -0.17778657376766205, -0.22231219708919525, 0.30085235834121704, -0.22739852964878082, 0.2308674305677414};

const float transformer_layers_0_1_fn_net_0_bias_weights_feedforward[24] = {0.07102790474891663, -0.10307249426841736, 0.10450132191181183, 0.06109493598341942, 0.14005491137504578, -0.23645825684070587, 0.18263697624206543, 0.18886609375476837, -0.2821384370326996, 0.012220816686749458, 0.19660378992557526, -0.2295934110879898, 0.03762337192893028, 0.06495407223701477, 0.10986986011266708, 0.19570238888263702, 0.1472238004207611, 0.18783628940582275, -0.03555670380592346, 0.15575765073299408, -0.211870014667511, -0.0939142182469368, -0.033838026225566864, -0.09595469385385513};

const float transformer_layers_0_1_fn_net_3_weight_weights_feedforward[288] = {0.13944563269615173, -0.053525347262620926, -0.16145893931388855, 0.02831614390015602, -0.1474788933992386, 0.05273429676890373, -0.00039892521454021335, -0.013576175086200237, 0.14792728424072266, 0.09587658196687698, -0.02482910268008709, 0.14489799737930298, 0.09319787472486496, 0.1386914849281311, 0.17637969553470612, 0.15977033972740173, -0.14774663746356964, -0.12392476201057434, -0.06346900761127472, -0.06937797367572784, -0.12660016119480133, -0.240245521068573, 0.10598506033420563, -0.13657623529434204, 0.10642680525779724, -0.021439919248223305, -0.0853416845202446, -0.09130174666643143, 0.022429879754781723, 0.08061377704143524, -0.07838930934667587, -0.03857889771461487, 0.048188839107751846, 0.0107615627348423, -0.019509075209498405, 0.051550038158893585, 0.12426739931106567, 0.11774994432926178, -0.0891275405883789, 0.20877870917320251, 0.0017775952583178878, 0.028345424681901932, -0.04682181030511856, -0.012096172198653221, 0.0042663938365876675, 0.07531093060970306, -0.052093300968408585, 0.11574655771255493, -0.06496056169271469, 0.10628154128789902, -0.03208819776773453, 0.153485968708992, -0.09260906279087067, 0.07626789808273315, -0.12670426070690155, -0.04325636476278305, 0.10268265753984451, -0.04772430285811424, 0.13357144594192505, -0.03824389725923538, 0.014231697656214237, 0.030805103480815887, -0.13477995991706848, 0.16825851798057556, 0.02760346606373787, -0.06157428398728371, 0.0346328429877758, -0.17769338190555573, 0.13709880411624908, -0.10965471714735031, 0.11589477211236954, 0.10239148139953613, 0.008135054260492325, 0.055287353694438934, 0.031674906611442566, -0.02260918915271759, -0.14497895538806915, -0.14778739213943481, 0.19839569926261902, 0.007764177862554789, -0.1834944635629654, 0.04365433007478714, 0.0882105901837349, -0.07614830881357193, 0.027364306151866913, -0.1484210342168808, -0.1637508124113083, 0.14857381582260132, 0.016058027744293213, 0.08183251321315765, 0.08235930651426315, 0.12683172523975372, -0.1488664299249649, 0.026745492592453957, -0.02337292581796646, -0.14932097494602203, 0.0922546461224556, -0.1219341978430748, 0.12520650029182434, -0.0353238619863987, 0.15791316330432892, 0.04633556306362152, -0.07859247177839279, 0.0824749693274498, -0.13415558636188507, 0.10628734529018402, 0.08658160269260406, -0.07580728828907013, -0.09070804715156555, -0.037188392132520676, 0.05531472712755203, -0.04249343276023865, -0.023678269237279892, -0.03923522308468819, -0.18681688606739044, -0.009739971719682217, -0.029381191357970238, -0.13214068114757538, 0.07093433290719986, 0.11639061570167542, -0.08254024386405945, 0.08834042400121689, -0.07723197340965271, 0.052382610738277435, 0.02738683670759201, 0.10698612034320831, -0.16460876166820526, 0.1577206254005432, 0.1530691534280777, 0.09869354218244553, -0.1945236474275589, -0.04589686542749405, 0.11960308998823166, -0.06126738712191582, -0.0773242637515068, 0.03787412866950035, -0.03397050499916077, -0.056629642844200134, -0.013240558095276356, -0.12486042082309723, 0.07022783905267715, -0.038191523402929306, -0.052919793874025345, 0.05007268488407135, -0.015810905024409294, -0.06769594550132751, -0.007002965081483126, 0.08806813508272171, 0.16373732686042786, 0.08254382014274597, 0.1087440773844719, -0.03748141974210739, 0.18798331916332245, 0.044476937502622604, -0.03432883322238922, 0.12742376327514648, -0.0792016014456749, -0.14693522453308105, -0.10652867704629898, 0.15013429522514343, 0.14378976821899414, -0.05959424749016762, -0.058431558310985565, -0.04135127738118172, 0.1406697928905487, 0.04983452707529068, -0.06850617378950119, 0.026328755542635918, 0.11965158581733704, 0.11514261364936829, 0.1398259401321411, -0.08434358239173889, -0.07080794870853424, 0.022467687726020813, -0.07582298666238785, -0.10271526128053665, -0.015310698188841343, -0.1747029572725296, 0.1765935719013214, -0.004957112018018961, 0.07622106373310089, 0.030533717945218086, 0.012050806544721127, 0.09005708247423172, -0.005811765789985657, -0.03321829065680504, -0.12271901220083237, 0.12332340329885483, -0.05387738719582558, 0.12338943034410477, -0.06691544502973557, 0.07103084772825241, -0.048710502684116364, -0.14117112755775452, -0.07349654287099838, 0.049007222056388855, -0.0739983394742012, 0.10717159509658813, -0.03373429924249649, -0.06917043775320053, 0.18260037899017334, -0.04871438816189766, 0.10324297100305557, -0.07361064106225967, -0.0663766860961914, -0.159317746758461, -0.005085009150207043, 0.09775704145431519, 0.16850116848945618, 0.10753359645605087, -0.16652241349220276, 0.024080945178866386, 0.1452803760766983, 0.17222274839878082, -0.016470281407237053, -0.13343149423599243, -0.16557928919792175, 0.09593328833580017, -0.03431267291307449, 0.0909319594502449, 0.03846234083175659, 0.10473548620939255, -0.14919236302375793, -0.016982203349471092, 0.07582364231348038, 0.057826846837997437, -0.05995234474539757, -0.0856255516409874, -0.15473920106887817, 0.007690732832998037, 0.12966443598270416, -0.06786919385194778, 0.1540103703737259, -0.00013011941337026656, 0.1424083411693573, -0.13048860430717468, -0.06958375871181488, -0.009115508757531643, -0.0930883064866066, -0.12195726484060287, 0.07065463811159134, -0.008060701191425323, -0.09656635671854019, -0.12134918570518494, 0.08926928043365479, 0.12010828405618668, -0.04759308695793152, -0.04868296906352043, 0.11280795186758041, -0.06423298269510269, -0.03969673067331314, -0.00044918336789123714, -0.03848082944750786, 0.19319839775562286, 0.15870998799800873, 0.0815899446606636, 0.03596747666597366, -0.18084114789962769, -0.06920889019966125, -0.16869162023067474, -0.053354643285274506, -0.3600388765335083, 0.008166614919900894, 0.13862557709217072, 0.12629035115242004, -0.11159496754407883, 0.09090818464756012, -0.06118348613381386, 0.20007239282131195, -0.10991839319467545, -0.07132891565561295, 0.13546858727931976, 0.1612650454044342, 0.010124449618160725, 0.025983689352869987, -0.06291642040014267, -0.12369105219841003, -0.11731840670108795, -0.1548750251531601, 0.0674407035112381, -0.0795009434223175, 0.027315443381667137, -0.13318197429180145, 0.07246241718530655, 0.08259894698858261, -0.11804705858230591, -0.022741714492440224, -0.15407556295394897};

const float transformer_layers_0_1_fn_net_3_bias_weights_feedforward[12] = {0.07433619350194931, -0.11970075964927673, -0.03243258222937584, -0.132437065243721, -0.08772770315408707, 0.06822407990694046, -0.15333791077136993, -0.06895110756158829, 0.11687694489955902, -0.07306504994630814, -0.06413431465625763, -0.03699713945388794};

const float mlp_head_0_weight_weights_layer[12] = {0.7165003418922424, 0.7512685060501099, 0.7252449989318848, 0.7571440935134888, 0.769547700881958, 0.7399262189865112, 0.7884064316749573, 0.7322722673416138, 0.7241777777671814, 0.4866485595703125, 0.8341264724731445, 0.7870186567306519};

const float mlp_head_0_bias_weights_layer[12] = {0.029226750135421753, 0.010225432924926281, 0.017041465267539024, 0.042638108134269714, -0.008274685591459274, -0.002369006397202611, -0.011329361237585545, -0.017400914803147316, 0.0009381461422890425, -0.2659428119659424, -0.005888660438358784, 0.019164051860570908};

const float mlp_head_1_weight_weights_layer[24] = {0.10717733204364777, 0.13304010033607483, -0.14490029215812683, -0.03990550711750984, 0.06238476186990738, 0.06475356221199036, 0.06480207294225693, 0.10847939550876617, -0.0028482696507126093, -0.10226079076528549, -0.02072426676750183, 0.04550180211663246, -0.1865699291229248, 0.23830336332321167, 0.08820535242557526, 0.01340116374194622, 0.03622233122587204, -0.1301712840795517, -0.017014402896165848, -0.02710609696805477, -0.24290280044078827, 0.16964249312877655, 0.1100742518901825, -0.09371419996023178};

const float mlp_head_1_bias_weights_layer[2] = {-0.10691665858030319, 0.17366425693035126};

