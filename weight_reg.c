uint8_t weight_layer1_conv1[288] = 
{0xbf,
0x3f,
0xe5,
0xaf,
0x41,
0xce,
0x9e,
0x3f,
0x6a,
0x2f,
0x27,
0xee,
0xd4,
0x37,
0xfc,
0x17,
0x93,
0x67,
0xdd,
0x89,
0x63,
0x5f,
0xc1,
0x47,
0xdd,
0xaf,
0xcf,
0x9e,
0x5b,
0x7f,
0x7c,
0xb6,
0xbc,
0x28,
0xb1,
0x3e,
0x2c,
0xb8,
0xe4,
0xf8,
0xe1,
0xd8,
0x58,
0xa7,
0x4b,
0x97,
0x6f,
0x50,
0xd4,
0x23,
0xb4,
0x48,
0xd1,
0x48,
0xff,
0x0f,
0x74,
0x03,
0x53,
0x4d,
0x74,
0x03,
0xb4,
0x67,
0xc9,
0xef,
0xa4,
0x58,
0xcd,
0xf1,
0x09,
0xf0,
0xa1,
0xf8,
0xf5,
0xc8,
0x45,
0x8c,
0xff,
0xd4,
0x7f,
0xf9,
0x35,
0xac,
0x4f,
0xb7,
0x2f,
0x37,
0x2e,
0x27,
0xcf,
0x48,
0x63,
0x96,
0x07,
0x70,
0xb1,
0x23,
0xbd,
0x6a,
0xb1,
0xba,
0x38,
0x2f,
0x98,
0xa5,
0x7c,
0x8d,
0x57,
0xc5,
0x50,
0xc7,
0x10,
0x87,
0x6b,
0x5f,
0x2e,
0x7f,
0xb2,
0x75,
0xc4,
0x5e,
0x84,
0x7e,
0xeb,
0xae,
0x08,
0xfd,
0x08,
0xbc,
0xa6,
0x9c,
0x88,
0x21,
0x80,
0xb8,
0xe4,
0x98,
0x49,
0x61,
0xee,
0x45,
0xf7,
0x44,
0xb8,
0x6c,
0x48,
0x97,
0x5e,
0x73,
0x68,
0xb3,
0x94,
0x68,
0xe1,
0x8b,
0x29,
0xc2,
0xc3,
0xf5,
0x6b,
0x58,
0x5f,
0x80,
0x7c,
0xeb,
0xd8,
0xa6,
0xd4,
0xc8,
0x49,
0x9b,
0x1f,
0xf3,
0x46,
0xb9,
0x6b,
0x1a,
0x91,
0xf6,
0xca,
0x02,
0x98,
0x2a,
0xe1,
0x48,
0x38,
0x83,
0x9c,
0x29,
0xf4,
0x48,
0x72,
0x91,
0x5e,
0xe1,
0x66,
0x51,
0x0a,
0x3c,
0x82,
0xbc,
0xde,
0x1d,
0x08,
0x22,
0x94,
0xa0,
0xf4,
0x89,
0x7c,
0x03,
0x3c,
0xe7,
0x7c,
0x23,
0x6a,
0xb0,
0x1c,
0x60,
0x1c,
0xaa,
0x21,
0x32,
0xb6,
0x73,
0xb0,
0x69,
0x2b,
0x11,
0xa6,
0xf2,
0xba,
0xf8,
0x47,
0xc8,
0x58,
0x81,
0x36,
0x01,
0xb6,
0xf8,
0xdb,
0x99,
0xd5,
0x01,
0xb0,
0xfc,
0xf0,
0xec,
0xdc,
0xcc,
0x18,
0xaf,
0x3d,
0x27,
0x96,
0x6a,
0xf7,
0x9d,
0xb3,
0x37,
0xb6,
0xe5,
0x86,
0x78,
0x81,
0x46,
0x94,
0xc1,
0xc7,
0x54,
0x42,
0x52,
0x9e,
0x67,
0xe7,
0x50,
0xa1,
0xc8,
0x48,
0x8b,
0xed,
0x08,
0xa7,
0xa8,
0x78,
0xae};

  // channel  0      tensor([-5., 10., 25., 40., 55., 69., 84.], device='cuda:0')
  hard_lut_0[0] <= -8'd5;
  hard_lut_0[1] <= 8'd10;
  hard_lut_0[2] <= 8'd25;
  hard_lut_0[3] <= 8'd40;
  hard_lut_0[4] <= 8'd55;
  hard_lut_0[5] <= 8'd69;
  hard_lut_0[6] <= 8'd84;
  // channel  1      tensor([-77., -63., -49., -35., -21.,  -8.,   6.], device='cuda:0')
  hard_lut_1[0] <= -8'd77;
  hard_lut_1[1] <= -8'd63;
  hard_lut_1[2] <= -8'd49;
  hard_lut_1[3] <= -8'd35;
  hard_lut_1[4] <= -8'd21;
  hard_lut_1[5] <= -8'd8;
  hard_lut_1[6] <= 8'd6;
  // channel  2      tensor([-12.,  -0.,  11.,  23.,  34.,  46.,  57.], device='cuda:0')
  hard_lut_2[0] <= -8'd12;
  hard_lut_2[1] <= 8'd0;
  hard_lut_2[2] <= 8'd11;
  hard_lut_2[3] <= 8'd23;
  hard_lut_2[4] <= 8'd34;
  hard_lut_2[5] <= 8'd46;
  hard_lut_2[6] <= 8'd57;
  // channel  3      tensor([-14.,  -2.,   9.,  20.,  32.,  43.,  54.], device='cuda:0')
  hard_lut_3[0] <= -8'd14;
  hard_lut_3[1] <= -8'd2;
  hard_lut_3[2] <= 8'd9;
  hard_lut_3[3] <= 8'd20;
  hard_lut_3[4] <= 8'd32;
  hard_lut_3[5] <= 8'd43;
  hard_lut_3[6] <= 8'd54;
  // channel  4      tensor([  4.,  20.,  36.,  52.,  68.,  84., 100.], device='cuda:0')
  hard_lut_4[0] <= 8'd4;
  hard_lut_4[1] <= 8'd20;
  hard_lut_4[2] <= 8'd36;
  hard_lut_4[3] <= 8'd52;
  hard_lut_4[4] <= 8'd68;
  hard_lut_4[5] <= 8'd84;
  hard_lut_4[6] <= 8'd100;
  // channel  5      tensor([ 6., 21., 37., 53., 68., 84., 99.], device='cuda:0')
  hard_lut_5[0] <= 8'd6;
  hard_lut_5[1] <= 8'd21;
  hard_lut_5[2] <= 8'd37;
  hard_lut_5[3] <= 8'd53;
  hard_lut_5[4] <= 8'd68;
  hard_lut_5[5] <= 8'd84;
  hard_lut_5[6] <= 8'd99;
  // channel  6      tensor([ 0., 16., 31., 46., 61., 77., 92.], device='cuda:0')
  hard_lut_6[0] <= 8'd0;
  hard_lut_6[1] <= 8'd16;
  hard_lut_6[2] <= 8'd31;
  hard_lut_6[3] <= 8'd46;
  hard_lut_6[4] <= 8'd61;
  hard_lut_6[5] <= 8'd77;
  hard_lut_6[6] <= 8'd92;
  // channel  7      tensor([-10.,   6.,  22.,  38.,  55.,  71.,  87.], device='cuda:0')
  hard_lut_7[0] <= -8'd10;
  hard_lut_7[1] <= 8'd6;
  hard_lut_7[2] <= 8'd22;
  hard_lut_7[3] <= 8'd38;
  hard_lut_7[4] <= 8'd55;
  hard_lut_7[5] <= 8'd71;
  hard_lut_7[6] <= 8'd87;
  // channel  8      tensor([-24., -12.,   0.,  12.,  24.,  36.,  49.], device='cuda:0')
  hard_lut_8[0] <= -8'd24;
  hard_lut_8[1] <= -8'd12;
  hard_lut_8[2] <= 8'd0;
  hard_lut_8[3] <= 8'd12;
  hard_lut_8[4] <= 8'd24;
  hard_lut_8[5] <= 8'd36;
  hard_lut_8[6] <= 8'd49;
  // channel  9      tensor([14., 25., 35., 46., 57., 68., 78.], device='cuda:0')
  hard_lut_9[0] <= 8'd14;
  hard_lut_9[1] <= 8'd25;
  hard_lut_9[2] <= 8'd35;
  hard_lut_9[3] <= 8'd46;
  hard_lut_9[4] <= 8'd57;
  hard_lut_9[5] <= 8'd68;
  hard_lut_9[6] <= 8'd78;
  // channel  10      tensor([ 5., 14., 23., 32., 41., 50., 59.], device='cuda:0')
  hard_lut_10[0] <= 8'd5;
  hard_lut_10[1] <= 8'd14;
  hard_lut_10[2] <= 8'd23;
  hard_lut_10[3] <= 8'd32;
  hard_lut_10[4] <= 8'd41;
  hard_lut_10[5] <= 8'd50;
  hard_lut_10[6] <= 8'd59;
  // channel  11      tensor([ 6., 20., 34., 48., 62., 76., 90.], device='cuda:0')
  hard_lut_11[0] <= 8'd6;
  hard_lut_11[1] <= 8'd20;
  hard_lut_11[2] <= 8'd34;
  hard_lut_11[3] <= 8'd48;
  hard_lut_11[4] <= 8'd62;
  hard_lut_11[5] <= 8'd76;
  hard_lut_11[6] <= 8'd90;
  // channel  12      tensor([-6.,  5., 17., 28., 40., 51., 62.], device='cuda:0')
  hard_lut_12[0] <= -8'd6;
  hard_lut_12[1] <= 8'd5;
  hard_lut_12[2] <= 8'd17;
  hard_lut_12[3] <= 8'd28;
  hard_lut_12[4] <= 8'd40;
  hard_lut_12[5] <= 8'd51;
  hard_lut_12[6] <= 8'd62;
  // channel  13      tensor([-2., 12., 26., 40., 55., 69., 83.], device='cuda:0')
  hard_lut_13[0] <= -8'd2;
  hard_lut_13[1] <= 8'd12;
  hard_lut_13[2] <= 8'd26;
  hard_lut_13[3] <= 8'd40;
  hard_lut_13[4] <= 8'd55;
  hard_lut_13[5] <= 8'd69;
  hard_lut_13[6] <= 8'd83;
  // channel  14      tensor([-22.,  -9.,   3.,  16.,  29.,  42.,  55.], device='cuda:0')
  hard_lut_14[0] <= -8'd22;
  hard_lut_14[1] <= -8'd9;
  hard_lut_14[2] <= 8'd3;
  hard_lut_14[3] <= 8'd16;
  hard_lut_14[4] <= 8'd29;
  hard_lut_14[5] <= 8'd42;
  hard_lut_14[6] <= 8'd55;
  // channel  15      tensor([-8.,  7., 22., 36., 51., 66., 81.], device='cuda:0')
  hard_lut_15[0] <= -8'd8;
  hard_lut_15[1] <= 8'd7;
  hard_lut_15[2] <= 8'd22;
  hard_lut_15[3] <= 8'd36;
  hard_lut_15[4] <= 8'd51;
  hard_lut_15[5] <= 8'd66;
  hard_lut_15[6] <= 8'd81;



  {0x7012002030740120,
0x7010002130740430,
0x7010001230740430,
0x7011000230740420,
0x7011001130740530,
0x7010002030740631,
0x7010000230740530,
0x7011001130740531,
0x7011001040740631,
0x7010001140740530,
0x7010001240740531,
0x7011001130740531,
0x7011001030740530,
0x7010001040740530,
0x7010001140740430,
0x7010000140740430,
0x7010000140740531,
0x7010000140740631,
0x7010001140740631,
0x7011001230730631,
0x7011001230730631,
0x7011001130730630,
0x7011001230730530,
0x7011001130730530,
0x7011000230730630,
0x7011001130730731,
0x7011001120730731,
0x7011001130730640,
0x7011000130730731,
0x6011000230720732,
0x5111000231720732,
0x2010006020630745,
0x7214002321240010,
0x6112003331330410,
0x6011003331430410,
0x6013001421430310,
0x6113002431430410,
0x5102003331340400,
0x6002002431430410,
0x6102002331340410,
0x6103001531440410,
0x5102002331330610,
0x5102003331331510,
0x6112003321330310,
0x7113003320330310,
0x7112002431330310,
0x7112002331430310,
0x7112002341430510,
0x6102001342430610,
0x4102002332333700,
0x4202003331124600,
0x4213003631225300,
0x4213003531115200,
0x5213003521113100,
0x6113003521121100,
0x6113003321120200,
0x7113002521230110,
0x6003002321230300,
0x6103002322330400,
0x6103002321330410,
0x6103001432330510,
0x4102001332430610,
0x3102002332440610,
0x0000007430437715,
0x7114002231340010,
0x6112003341430410,
0x7011003331530410,
0x7002001322430510,
0x6112003221440610,
0x6002002431440510,
0x6002002331440510,
0x6102002331430410,
0x6102002331440510,
0x5002002431440510,
0x5001002622451510,
0x4003002112050700,
0x7203003221240610,
0x7212003531330410,
0x7113002332230410,
0x6103003332131400,
0x6103002532234500,
0x3203002331127600,
0x4313002732417300,
0x1213002423027700,
0x1213001614027700,
0x2313003303015700,
0x4313005302013510,
0x6212004712012110,
0x5213004211011300,
0x7112004710110010,
0x6013001522130100,
0x5103002322130300,
0x5103002332240410,
0x4103002432340510,
0x3103001432440510,
0x0000007330437715,
0x7114003221240010,
0x6102002331340410,
0x7002004231340410,
0x7002002421430410,
0x6102003330330510,
0x6012002431430410,
0x6002002331330400,
0x6103002431430310,
0x6102002331430410,
0x5001003430440510,
0x5000006620470530,
0x2005000102073700,
0x7406002522251400,
0x6413005331230500,
0x6103003521034300,
0x5203003421026300,
0x5313004520107200,
0x4314000723507000,
0x0112003213125703,
0x0003002402046701,
0x0103013502046700,
0x0204011603045500,
0x1304012504043600,
0x1303004103032700,
0x4313003704023410,
0x3312006002010710,
0x7112005511020230,
0x6012003521130010,
0x6003002422130200,
0x5103002331330400,
0x5103002431440410,
0x0000006430537715,
0x7214004221040000,
0x5103003531033200,
0x7112003631131000,
0x7013002422130200,
0x6113002431230400,
0x6112003231330510,
0x6012001431430410,
0x6113002331430510,
0x6112002331430410,
0x5011004240330510,
0x4010007040131420,
0x7004200730727000,
0x4407100422331000,
0x3515004131023300,
0x4313004520107100,
0x5314000713307000,
0x2414000605006601,
0x1212005003013732,
0x0000007300044722,
0x1001013600145410,
0x0002013500254400,
0x0004021500352300,
0x0105022300150100,
0x1204011702351000,
0x0204013002050600,
0x2304012501031200,
0x5412005411010200,
0x6111005521020010,
0x6012002421130110,
0x5003001322230300,
0x5103002222340510,
0x0000006430537715,
0x7314003611030100,
0x1414002221007500,
0x7422006730201010,
0x7022002631020000,
0x6013001232030200,
0x6112003331330410,
0x6012002431430510,
0x6103001431430510,
0x6102003231340510,
0x5111002441430510,
0x2320005440404702,
0x6132000422703750,
0x4225000611411620,
0x1315000513125701,
0x0514000515117701,
0x1412004204014731,
0x1104003303033720,
0x0100007300042511,
0x1000007700242310,
0x1000011400240610,
0x0002012200340600,
0x2103012300640300,
0x2104010700740000,
0x1003013000350400,
0x0104011400250200,
0x2305010703431200,
0x0404013113020600,
0x5310007210010220,
0x7010005510120030,
0x6003000411130110,
0x5103002321340300,
0x1000006420537714,
0x3304003312031700,
0x0507000706207700,
0x2721007013010712,
0x7020007220020350,
0x7012002531030020,
0x7013002431330200,
0x6112003331430410,
0x6002002431440410,
0x6103002331340400,
0x5102003331232500,
0x4211002741316500,
0x4212001523415720,
0x2012003211133721,
0x1005000502145700,
0x0205001503356600,
0x0202016210153602,
0x0003011611252300,
0x0201014310141401,
0x2010007210330320,
0x3000012700440120,
0x1004010400440300,
0x2304010200630500,
0x3212003200630520,
0x2002014200430300,
0x3104010700630100,
0x0205010101330600,
0x0305010712431100,
0x2302005012030300,
0x7110007310020130,
0x7012001611120020,
0x6004001322230300,
0x1100006420437714,
0x0106000201054700,
0x1407020605547600,
0x0402007502160204,
0x2000007001050700,
0x7012003521030120,
0x7113002531330210,
0x6112003331230510,
0x6102003331231510,
0x6002002622243510,
0x2204001222037700,
0x4313002723427510,
0x1001005212144712,
0x0001005200054700,
0x2005000701345400,
0x1205011310444300,
0x0202015310442201,
0x1002011510440200,
0x1103011511340200,
0x1210007010120401,
0x6020006610610040,
0x4014010700530000,
0x1206010201340500,
0x1303005000330601,
0x3211002600520310,
0x2103010401520520,
0x0104010502430501,
0x0204012002340711,
0x1303013412230100,
0x5210007210020110,
0x7010003611120030,
0x6004000422030210,
0x1100006430436714,
0x1006010701356300,
0x0407020205464700,
0x0200017000070414,
0x5002001700070000,
0x7205000212050000,
0x7313004331240210,
0x5102003431231400,
0x5000003723354410,
0x1002003203076701,
0x1007000706077700,
0x0505007003075701,
0x3100007500365620,
0x2001002600246300,
0x3105010700634200,
0x1306010301542400,
0x0302014200430602,
0x1102012200330610,
0x2103012400420210,
0x2211012710520000,
0x3020007000220733,
0x4011000300220740,
0x3006010700440300,
0x0305010302341501,
0x1303003102330711,
0x2102014300330620,
0x1003011501440520,
0x0104011402450401,
0x1205010304340500,
0x2401007011020400,
0x7120007310110040,
0x7012000721120020,
0x1000004320235703,
0x1006020500135600,
0x2207021700762000,
0x0000017000070405,
0x7003010700150000,
0x7207000411150000,
0x7414004331340100,
0x3202003230123600,
0x4010007530355341,
0x0000005500077200,
0x4007010700177000,
0x3307010700476000,
0x3301007200563200,
0x3003000600434500,
0x3306000200531600,
0x3206010601640410,
0x0204013000240702,
0x2202012701631310,
0x0204010202320600,
0x3211005401530330,
0x2010007300050524,
0x2000000300040710,
0x3004002500450620,
0x1107010300251400,
0x2305012500541200,
0x1102013401440401,
0x0003011200240610,
0x1103012410440310,
0x1105010502450200,
0x1304011504240200,
0x3410007011020510,
0x7020005510110140,
0x3010004720324311,
0x0007020702317700,
0x1412017000440522,
0x1000025700250000,
0x5014010400140000,
0x7217000302240000,
0x7414006020350200,
0x4104000622533700,
0x0320007030017706,
0x6030005710707030,
0x3025210500212000,
0x6217100600530000,
0x2412007000340402,
0x4014000700541510,
0x1207000100240700,
0x3304005100440510,
0x2105010701641200,
0x0203013001240703,
0x1206020602431400,
0x2200007210450334,
0x1000007100070523,
0x2003010700053400,
0x3303014300441300,
0x3207010700532100,
0x2405012301530200,
0x1101016100240402,
0x2002010700440100,
0x0104011101240601,
0x1104010402440410,
0x1206010403250300,
0x1302005402040301,
0x4210006001020520,
0x5010007710420450,
0x1007130702417700,
0x0720017000530305,
0x2020024500230410,
0x5015010300030000,
0x7217000501340000,
0x7315006020350000,
0x6004000522740410,
0x0510004420607707,
0x1150007000705776,
0x5031100700300630,
0x7227100300410100,
0x4221007600530001,
0x3012001300250610,
0x2007010601351500,
0x1505003100450700,
0x3104002300540620,
0x0104011601451401,
0x0207020204341700,
0x1200007010360436,
0x1000007200050200,
0x5004020700533000,
0x1515013100331300,
0x3317010702621300,
0x0405013101330502,
0x0201015300230501,
0x2001013300330420,
0x1004010610440100,
0x0104011301440301,
0x0105020402350300,
0x1203014302160200,
0x3202002402040200,
0x3010007110330742,
0x1007130703517700,
0x0720017000650407,
0x0010016100040710,
0x4016020300010200,
0x7527001700520000,
0x7215004220130000,
0x7005000222340310,
0x2204000623737702,
0x0320007010526717,
0x5020000700401730,
0x6225000701620410,
0x1120007000040714,
0x4020006600240630,
0x3007010700352400,
0x1507001300461400,
0x1304005000450600,
0x2003013400451310,
0x1107020703652100,
0x0300007010070307,
0x3010014700031200,
0x5015010600421600,
0x2517011400421400,
0x2317010201520600,
0x1304014300330301,
0x1202013400330300,
0x2111013310320410,
0x2002010601530320,
0x0003012200240602,
0x1004020401350410,
0x2103013300360200,
0x4105000513350100,
0x1100007020250615,
0x2007130704525700,
0x0610017000460307,
0x1010024500240400,
0x4117020703411300,
0x5727005001020400,
0x7323007410020020,
0x7014001521030110,
0x4106000523535700,
0x0410007120427707,
0x4020000700502630,
0x5011005200250530,
0x0000005300052500,
0x3120007100124600,
0x6016010700614000,
0x3317010600540000,
0x1205002200350301,
0x0203012100332700,
0x3207020602731320,
0x0210007110150007,
0x3010011600120410,
0x4016010600421500,
0x2417010600630300,
0x0206010202340700,
0x0304013101240701,
0x1202012601440510,
0x0102013201240711,
0x1001015100240730,
0x1001012500350410,
0x1004010300350400,
0x3203013210450100,
0x6004000522550020,
0x0100007120150505,
0x3007130703534610,
0x0510017000260307,
0x3010012700330300,
0x4007020703631500,
0x0707013001040500,
0x7422007310030010,
0x7013002411030110,
0x5107000724334600,
0x0400007220337707,
0x1010002200222730,
0x3010007400160531,
0x1001000700064600,
0x0414010000026700,
0x5534011700703720,
0x2225014000220510,
0x4114013500430000,
0x2105010700632000,
0x0206010102540700,
0x0210007010140506,
0x4020012700420320,
0x3015010700620400,
0x0316013000230700,
0x3205012600530000,
0x1205010610531000,
0x1102014210440201,
0x1002012500350100,
0x1001014200240300,
0x2001013400440310,
0x3004010410540100,
0x4203001420740000,
0x4003002020350420,
0x1000006530442503,
0x3007030702434710,
0x0511017000250204,
0x4020012600430400,
0x3007020500540710,
0x0507020301152500,
0x3611007100050100,
0x6013002601030000,
0x6107000713141300,
0x0302007030043703,
0x4010005720733030,
0x0010007000031603,
0x6020113700414000,
0x5117120700714000,
0x0526012000320703,
0x3323012600510200,
0x2223000300310500,
0x2114011300420620,
0x2005011310540210,
0x2110007310650104,
0x2010001000420720,
0x4012000600720530,
0x1115010700640300,
0x0305010102430701,
0x1305011101430710,
0x1211016200330622,
0x2000014400330320,
0x2001012400330100,
0x4012002210520100,
0x6014000511730020,
0x5003003020650320,
0x5002002120450310,
0x2000004530644712,
0x2006020701336700,
0x0512017100252302,
0x3020024400230400,
0x4015020500340410,
0x0307030702552300,
0x0712017000040303,
0x6011012700230010,
0x3006000001050400,
0x2306000712342200,
0x1310007020230734,
0x2020004710407100,
0x5240000700701600,
0x5146100000700750,
0x2326000500510201,
0x2222002210520611,
0x1113011200120700,
0x2212013510320410,
0x4003001501340210,
0x3110007010040501,
0x7120001720702010,
0x5130003720700030,
0x1014010200330400,
0x1106010210330200,
0x3305011710731000,
0x0202013210420202,
0x1110014300420311,
0x3011011600420110,
0x4013000201330310,
0x7013003001340540,
0x6002005010150320,
0x7002001621440210,
0x1100004430536713,
0x2004020700535610,
0x0414025200252402,
0x1121025300131400,
0x3023021500230400,
0x2107030702540300,
0x0705027000240404,
0x4120017000230430,
0x5004010700140000,
0x4107000503160200,
0x0501007010043604,
0x4220000700705510,
0x5230001701700640,
0x1023000000020721,
0x3114001501530730,
0x1101005100250722,
0x0104010401135700,
0x1302016300331510,
0x4001006300150020,
0x3004000702040000,
0x2414000607422700,
0x0420007000010736,
0x4020003710403440,
0x4216010712702000,
0x0415011002310703,
0x1211013501520422,
0x1010003100320722,
0x4010004210530540,
0x4002000310450330,
0x4002004010260310,
0x5002002520340100,
0x5103000531440310,
0x0100004430537704,
0x3011020700623630,
0x0214023300441401,
0x0222024400430400,
0x2122022400330400,
0x3007030500540400,
0x0507020300351200,
0x1410017000230402,
0x4011021500120220,
0x5006010700240000,
0x1205010702252000,
0x0406000004122700,
0x4320007000430751,
0x3003010700130200,
0x4002004500550110,
0x1000005300270402,
0x0007020702355700,
0x0604016000342701,
0x4220007410330020,
0x6004000700230010,
0x2006002210260300,
0x0303013610146101,
0x2413010613604600,
0x3314002003510740,
0x1204011602320412,
0x1000016201240623,
0x3002002301250310,
0x7000007010460330,
0x7000002120550330,
0x7002001420650110,
0x7002000431640310,
0x6102001231540510,
0x1000003530637714,
0x5020010700711540,
0x0014010200430600,
0x1222023200530500,
0x2022022400530400,
0x2017020400540500,
0x0307020501541300,
0x0510017100340405,
0x2010010701240720,
0x0105023000030700,
0x6314016600420020,
0x5107020703732000,
0x0410007000040607,
0x6011010701521130,
0x1010007000050713,
0x1000007100060600,
0x4007020701754300,
0x0507020302562502,
0x1510007000030701,
0x7021002610500040,
0x3113004410312000,
0x2215010710605000,
0x1306010413411201,
0x0304014001021603,
0x2102013602331320,
0x0000017000040303,
0x6002012700140010,
0x7001006210050000,
0x7002001520040000,
0x7103001521130200,
0x6213003532131300,
0x5203002532033400,
0x1100005530327713,
0x5030005700511540,
0x1014010400131600,
0x2223013300530300,
0x3121013300530400,
0x2115020400431500,
0x2307020500631200,
0x1310017400550004,
0x1000015400270421,
0x0007020704272700,
0x0707011005060701,
0x4406013304450760,
0x0201017601072703,
0x1000015002060740,
0x0000017100072700,
0x3000014700154100,
0x4007020500452500,
0x0307020501362300,
0x0505012502250300,
0x2513002003010700,
0x2522001702405720,
0x0323011303203731,
0x0204022403021732,
0x0103021704043702,
0x0101026002052713,
0x0100025701045600,
0x3101025201030510,
0x5212017600020000,
0x6213003712016000,
0x4316002713017200,
0x2414006512017400,
0x3414004713017200,
0x2111007610026200,
0x0030017200023722,
0x3013020700022300,
0x3224013200230300,
0x3122014300330400,
0x2116020600522600,
0x1417020502630600,
0x0210007000370727,
0x0000017000070720,
0x6005030700570020,
0x0007030700574000,
0x0507035000073700,
0x2302037700265000,
0x1000037600054000,
0x1010035700126000,
0x4122032700304000,
0x3126030600213000,
0x1207030600225000,
0x2407032600131000,
0x2307020703133000,
0x0305021205045703,
0x0203034203045703,
0x0102036301053711,
0x0002045500064210,
0x0002044700065000,
0x0102043600045200,
0x0302045500035300,
0x0302043701027300,
0x0314034602017600,
0x0405033702017500,
0x0504034702017500,
0x0404044602017700,
0x0212027300030000,
0x0430037100007605,
0x5031040700201010,
0x2025022400030000,
0x2123024300130200,
0x2016020600441600,
0x1107020301460600,
0x0000007000070506,
0x2010014700151200,
0x4022020300531500,
0x2217030000325000,
0x1617040710717000,
0x0624033110303002,
0x0330034400203100,
0x1341031700304200,
0x1442032300101400,
0x1343031600203410,
0x0335030701204600,
0x0425033200001601,
0x0205033301021510,
0x0104045600046200,
0x0103044700056000,
0x0102043500054100,
0x0101044400044300,
0x0102053600134200,
0x0102052600133200,
0x0102053500033401,
0x0102053400132500,
0x0102053500131200,
0x0103051600232200,
0x0203053400133401,
0x0103052500233400,
0x2111037200040000,
0x0620047200007707,
0x3140045400000330,
0x5031034700100010,
0x2022023400020000,
0x2016020100120200,
0x5014024700760010,
0x0010017000070004,
0x3010010700333300,
0x1335020200206600,
0x2747030700707300,
0x0737030003603703,
0x0524034200100704,
0x0322032401102712,
0x0322032301101721,
0x0322032401100721,
0x0322033201001722,
0x0213043300112731,
0x0203043500021500,
0x0203043500031400,
0x0102053500133501,
0x0102053400132401,
0x0102052500131200,
0x0102052500232301,
0x0102052400231501,
0x0001053400230501,
0x0002052400340501,
0x0101053300440501,
0x0101052400440501,
0x0102052300330500,
0x0102053400340401,
0x0002052500440301,
0x2112047000050000,
0x0400053500327707,
0x0420054200000700,
0x1240047000000630,
0x4030035700100020,
0x3026030700310000,
0x3130017000240522,
0x1020007300050300,
0x2022020500215700,
0x1737030700707300,
0x0737030602707700,
0x0417044000122706,
0x0203044400131401,
0x0102043400141400,
0x0103052400241300,
0x0103053400240200,
0x0102043500240301,
0x0102052400330401,
0x0102052400330401,
0x0102052500330401,
0x0101053300330502,
0x0001053400330502,
0x0102052500330501,
0x0102053300240501,
0x0001053500340401,
0x0001054200240502,
0x0002052400340300,
0x0002052400340300,
0x0102052400341401,
0x0102062500241400,
0x0102062600340401,
0x0002063200150702,
0x1103045000050000,
0x0102060600647707,
0x0303062400240200,
0x0310056300010300,
0x2130047100000410,
0x6033030700410040,
0x1020017000050103,
0x3020015600031300,
0x1226020700307600,
0x1737030703707700,
0x0727036000311707,
0x0103044600332413,
0x0102053300240401,
0x0102052400330300,
0x0102052500430200,
0x0102052400330301,
0x0102042400330401,
0x0102052400330501,
0x0102052400340502,
0x0002053300340502,
0x0001053400340402,
0x0001053400341502,
0x0001052400340501,
0x0002063400250401,
0x0001064200140501,
0x0101063500331100,
0x0102062500330100,
0x0102062400231200,
0x0101063600332201,
0x0002061500241402,
0x0002065000050602,
0x0001065300250200,
0x4105041700350000,
0x0102060500737707,
0x0204062400340200,
0x0202063500130100,
0x0310054400010400,
0x2120037000000530,
0x4030027700030010,
0x2020020700011100,
0x1327030702507600,
0x0727030002713701,
0x0404046200243707,
0x0101054200331502,
0x0002052600440200,
0x0103050400330401,
0x0102053200330501,
0x0102053400330301,
0x0002052400340301,
0x0002052400340401,
0x0002053300340402,
0x0002053400341401,
0x0001053500341401,
0x0001053300241502,
0x0101064400331301,
0x0001063500340100,
0x0002061500331101,
0x0102061400330400,
0x0202062300220501,
0x0112062600321501,
0x0000064200140713,
0x0001065100040601,
0x0101064400230000,
0x0112061700620000,
0x4125041200330000,
0x0102061500637707,
0x0203062400440301,
0x0103063400240200,
0x0202063500130100,
0x0310056300010300,
0x1240037000000310,
0x7040030700400020,
0x0117140602204500,
0x0717035110632404,
0x0203043400343303,
0x0102052500341302,
0x0101053200240502,
0x0003051500340400,
0x0103052400340501,
0x0102053200240501,
0x0102053400330300,
0x0102053400330301,
0x0001053500430301,
0x0002052400340402,
0x0001053300340402,
0x0002061600431301,
0x0101062300330602,
0x0101063300330501,
0x0102062500420400,
0x0102061500430301,
0x0102061500340502,
0x0001064200250714,
0x0000066100041602,
0x0101072700422000,
0x0313060700500000,
0x0221053300610311,
0x3023034000030010,
0x0102060300537707,
0x0203063300440401,
0x0102062500440201,
0x0102062300240200,
0x0202052600230100,
0x0430047100000502,
0x4150037100100750,
0x4037050701702020,
0x0616045010120707,
0x0212043410422301,
0x0002053500420101,
0x0001053400330202,
0x0002051500330200,
0x0003052400340201,
0x0103051500340201,
0x0102052400340401,
0x0102052400340602,
0x0001053200240712,
0x0001053400340511,
0x0101052500340401,
0x0102052300430602,
0x0102062400430402,
0x0101062400430301,
0x0101062400530401,
0x0001062400440512,
0x0001063300350613,
0x0000065100161603,
0x0001070700543300,
0x0203060400420701,
0x0315060201420700,
0x0310057000230523,
0x4011035400040010,
0x0002060700737705,
0x0204061200440302,
0x0101064300540302,
0x0002061400340301,
0x0102062300240300,
0x0211051700220200,
0x0430045100000702,
0x2024040401400770,
0x0315043500221504,
0x0202052300421602,
0x0111054200320612,
0x0110053300220411,
0x0111053400420310,
0x0001052500420100,
0x0003051500430000,
0x0002053300340101,
0x0002053400340201,
0x0001052500450201,
0x0002051500450402,
0x0002052300350602,
0x0102052300340601,
0x0102052400530401,
0x0102051400530402,
0x0001053300440513,
0x0000055200350512,
0x0000054400360412,
0x0001061500461401,
0x0103060300541600,
0x0205071300540300,
0x0205060700640000,
0x0301056000040404,
0x5020037400230020,
0x0102060500727706,
0x0104061300440301,
0x0101064200440403,
0x0002061400530301,
0x0102062400340301,
0x0103052400240300,
0x0202052600230100,
0x0102052100220710,
0x0103051500330400,
0x0202052300430402,
0x0101053300330402,
0x0101051500320501,
0x0101052400420612,
0x0111053200220712,
0x0111053300310420,
0x0011054500420011,
0x0000052500420101,
0x0001052400330302,
0x0001053300340301,
0x0002051500340200,
0x0103050500540200,
0x0103052300540402,
0x0102052200440402,
0x0000055300440312,
0x0000055300350302,
0x0000055300340200,
0x0002060600641100,
0x0205060400631300,
0x0305060500730201,
0x0204063000440412,
0x0103061600340101,
0x3120037000040010,
0x0001041000740727,
0x0002040200760522,
0x0001040100760715,
0x0000040100760734,
0x0001040100760724,
0x0001040200760723,
0x0002030100770723,
0x0001030100770723,
0x0002030200770733,
0x0002030100770724,
0x0000032000770724,
0x0001030200770733,
0x0001031000770734,
0x0001030300770723,
0x0001030101770725,
0x0000032000760745,
0x0000033000750743,
0x0000033100750743,
0x0000032200750732,
0x0001030400740521,
0x0002040200750622,
0x0002031000760714,
0x0001030300760723,
0x0000034000760724,
0x0000033200740522,
0x0010032300730411,
0x0011040400720420,
0x0014040400740513,
0x0004040000760726,
0x0001042000770726,
0x0002040300770732,
0x0100027000270401
};