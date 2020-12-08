/*
 *       /\        Matteo Cicuttin (C) 2016, 2017, 2018
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

#pragma once

namespace dunavant_quadratures {

static double rule_1[][4] = {
	{ 0.333333333333333, 0.333333333333333, 0.333333333333333,  1.000000000000000 }
};

static double rule_2[][4] = {
	{ 0.666666666666667, 0.166666666666667, 0.166666666666667,  0.333333333333333 },
	{ 0.166666666666667, 0.666666666666667, 0.166666666666667,  0.333333333333333 },
	{ 0.166666666666667, 0.166666666666667, 0.666666666666667,  0.333333333333333 }
};

static double rule_3[][4] = {
	{ 0.333333333333333, 0.333333333333333, 0.333333333333333, -0.562500000000000 },
    { 0.600000000000000, 0.200000000000000, 0.200000000000000,  0.520833333333333 },
    { 0.200000000000000, 0.600000000000000, 0.200000000000000,  0.520833333333333 },
    { 0.200000000000000, 0.200000000000000, 0.600000000000000,  0.520833333333333 }
};

static double rule_4[][4] = {
	{ 0.108103018168070, 0.445948490915965, 0.445948490915965,  0.223381589678011 },
	{ 0.445948490915965, 0.108103018168070, 0.445948490915965,  0.223381589678011 },
	{ 0.445948490915965, 0.445948490915965, 0.108103018168070,  0.223381589678011 },
    { 0.816847572980459, 0.091576213509771, 0.091576213509771,  0.109951743655322 },
    { 0.091576213509771, 0.816847572980459, 0.091576213509771,  0.109951743655322 },
    { 0.091576213509771, 0.091576213509771, 0.816847572980459,  0.109951743655322 }
};

static double rule_5[][4] = {
	{ 0.333333333333333, 0.333333333333333, 0.333333333333333,  0.225000000000000 },
    { 0.059715871789770, 0.470142064105115, 0.470142064105115,	0.132394152788506 },
    { 0.470142064105115, 0.059715871789770, 0.470142064105115,	0.132394152788506 },
    { 0.470142064105115, 0.470142064105115,	0.059715871789770,  0.132394152788506 },
    { 0.797426985353087, 0.101286507323456, 0.101286507323456,  0.125939180544827 },
    { 0.101286507323456, 0.797426985353087, 0.101286507323456,  0.125939180544827 },
    { 0.101286507323456, 0.101286507323456, 0.797426985353087,  0.125939180544827 },
};

static double rule_6[][4] = {
    { 0.501426509658179, 0.249286745170910, 0.249286745170910,  0.116786275726379 },
    { 0.249286745170910, 0.501426509658179, 0.249286745170910,  0.116786275726379 },
    { 0.249286745170910, 0.249286745170910, 0.501426509658179,  0.116786275726379 },
    { 0.873821971016996, 0.063089014491502, 0.063089014491502,  0.050844906370207 },
    { 0.063089014491502, 0.873821971016996, 0.063089014491502,  0.050844906370207 },
    { 0.063089014491502, 0.063089014491502, 0.873821971016996,  0.050844906370207 },
    { 0.053145049844817, 0.310352451033784, 0.636502499121399,  0.082851075618374 },
    { 0.053145049844817, 0.636502499121399, 0.310352451033784,  0.082851075618374 },
    { 0.310352451033784, 0.053145049844817, 0.636502499121399,  0.082851075618374 },
    { 0.310352451033784, 0.636502499121399, 0.053145049844817,  0.082851075618374 },
    { 0.636502499121399, 0.053145049844817, 0.310352451033784,  0.082851075618374 },
    { 0.636502499121399, 0.310352451033784, 0.053145049844817,  0.082851075618374 },
};

static double rule_7[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333, -0.149570044467682 },
    { 0.479308067841920, 0.260345966079040, 0.260345966079040,  0.175615257433208 },
    { 0.260345966079040, 0.479308067841920, 0.260345966079040,  0.175615257433208 },
    { 0.260345966079040, 0.260345966079040, 0.479308067841920,  0.175615257433208 },
    { 0.869739794195568, 0.065130102902216, 0.065130102902216,  0.053347235608838 },
    { 0.065130102902216, 0.869739794195568, 0.065130102902216,  0.053347235608838 },
    { 0.065130102902216, 0.065130102902216, 0.869739794195568,  0.053347235608838 },  
    { 0.048690315425316, 0.312865496004874, 0.638444188569810,  0.077113760890257 },
    { 0.048690315425316, 0.638444188569810, 0.312865496004874,  0.077113760890257 },
    { 0.312865496004874, 0.048690315425316, 0.638444188569810,  0.077113760890257 },
    { 0.312865496004874, 0.638444188569810, 0.048690315425316,  0.077113760890257 },
    { 0.638444188569810, 0.048690315425316, 0.312865496004874,  0.077113760890257 },
    { 0.638444188569810, 0.312865496004874, 0.048690315425316,  0.077113760890257 }
};

static double rule_8[][4] = {
	{ 0.333333333333333, 0.333333333333333, 0.333333333333333,  0.144315607677787 },
    { 0.081414823414554, 0.459292588292723, 0.459292588292723,  0.095091634267285 },
    { 0.459292588292723, 0.081414823414554, 0.459292588292723,  0.095091634267285 },
    { 0.459292588292723, 0.459292588292723, 0.081414823414554,  0.095091634267285 },
    { 0.658861384496480, 0.170569307751760, 0.170569307751760,  0.103217370534718 },
    { 0.170569307751760, 0.658861384496480, 0.170569307751760,  0.103217370534718 },
    { 0.170569307751760, 0.170569307751760, 0.658861384496480,  0.103217370534718 },
    { 0.898905543365938, 0.050547228317031, 0.050547228317031,  0.032458497623198 },
    { 0.050547228317031, 0.898905543365938, 0.050547228317031,  0.032458497623198 },
    { 0.050547228317031, 0.050547228317031, 0.898905543365938,  0.032458497623198 },
    { 0.008394777409958, 0.263112829634638, 0.728492392955404,  0.027230314174435 },
    { 0.008394777409958, 0.728492392955404, 0.263112829634638,  0.027230314174435 },
    { 0.263112829634638, 0.008394777409958, 0.728492392955404,  0.027230314174435 },
    { 0.263112829634638, 0.728492392955404, 0.008394777409958,  0.027230314174435 },
    { 0.728492392955404, 0.008394777409958, 0.263112829634638,  0.027230314174435 },
    { 0.728492392955404, 0.263112829634638, 0.008394777409958,  0.027230314174435 }
};

static double rule_9[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.097135796282799 }, //
    { 0.020634961602525, 0.489682519198738, 0.489682519198738, 0.031334700227139 }, //
    { 0.489682519198738, 0.020634961602525, 0.489682519198738, 0.031334700227139 },
    { 0.489682519198738, 0.489682519198738, 0.020634961602525, 0.031334700227139 },
    { 0.125820817014127, 0.437089591492937, 0.437089591492937, 0.077827541004774 }, //
    { 0.437089591492937, 0.125820817014127, 0.437089591492937, 0.077827541004774 },
    { 0.437089591492937, 0.437089591492937, 0.125820817014127, 0.077827541004774 },
    { 0.623592928761935, 0.188203535619033, 0.188203535619033, 0.079647738927210 }, //
    { 0.188203535619033, 0.623592928761935, 0.188203535619033, 0.079647738927210 },
    { 0.188203535619033, 0.188203535619033, 0.623592928761935, 0.079647738927210 },
    { 0.910540973211095, 0.044729513394453, 0.044729513394453, 0.025577675658698 }, //
    { 0.044729513394453, 0.910540973211095, 0.044729513394453, 0.025577675658698 },
    { 0.044729513394453, 0.044729513394453, 0.910540973211095, 0.025577675658698 },
    { 0.036838412054736, 0.221962989160766, 0.741198598784498, 0.043283539377289 }, //
    { 0.036838412054736, 0.741198598784498, 0.221962989160766, 0.043283539377289 },
    { 0.221962989160766, 0.036838412054736, 0.741198598784498, 0.043283539377289 },
    { 0.221962989160766, 0.741198598784498, 0.036838412054736, 0.043283539377289 },
    { 0.741198598784498, 0.036838412054736, 0.221962989160766, 0.043283539377289 },
    { 0.741198598784498, 0.221962989160766, 0.036838412054736, 0.043283539377289 }
};

static double rule_10[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.090817990382754 }, //
    
    { 0.028844733232685, 0.485577633383657, 0.485577633383657, 0.036725957756467 }, //
    { 0.485577633383657, 0.028844733232685, 0.485577633383657, 0.036725957756467 },
    { 0.485577633383657, 0.485577633383657, 0.028844733232685, 0.036725957756467 },
    
    { 0.781036849029926, 0.109481575485037, 0.109481575485037, 0.045321059435528 }, //
    { 0.109481575485037, 0.781036849029926, 0.109481575485037, 0.045321059435528 },
    { 0.109481575485037, 0.109481575485037, 0.781036849029926, 0.045321059435528 },
    
    { 0.141707219414880, 0.307939838764121, 0.550352941820999, 0.072757916845420 }, //
    { 0.141707219414880, 0.550352941820999, 0.307939838764121, 0.072757916845420 },
    { 0.307939838764121, 0.141707219414880, 0.550352941820999, 0.072757916845420 },
    { 0.307939838764121, 0.550352941820999, 0.141707219414880, 0.072757916845420 },
    { 0.550352941820999, 0.141707219414880, 0.307939838764121, 0.072757916845420 },
    { 0.550352941820999, 0.307939838764121, 0.141707219414880, 0.072757916845420 },
    
    { 0.025003534762686, 0.246672560639903, 0.728323904597411, 0.028327242531057 }, //
    { 0.025003534762686, 0.728323904597411, 0.246672560639903, 0.028327242531057 },
    { 0.246672560639903, 0.025003534762686, 0.728323904597411, 0.028327242531057 },
    { 0.246672560639903, 0.728323904597411, 0.025003534762686, 0.028327242531057 },
    { 0.728323904597411, 0.025003534762686, 0.246672560639903, 0.028327242531057 },
    { 0.728323904597411, 0.246672560639903, 0.025003534762686, 0.028327242531057 },
    
    { 0.009540815400299, 0.066803251012200, 0.923655933587500, 0.009421666963733 }, //
    { 0.009540815400299, 0.923655933587500, 0.066803251012200, 0.009421666963733 },
    { 0.066803251012200, 0.009540815400299, 0.923655933587500, 0.009421666963733 },
    { 0.066803251012200, 0.923655933587500, 0.009540815400299, 0.009421666963733 },
    { 0.923655933587500, 0.009540815400299, 0.066803251012200, 0.009421666963733 },
    { 0.923655933587500, 0.066803251012200, 0.009540815400299, 0.009421666963733 }
};

static double rule_11[][4] = {
    { -0.069222096541517, 0.534611048270758, 0.534611048270758, 0.000927006328961 }, //
    { 0.534611048270758, -0.069222096541517, 0.534611048270758, 0.000927006328961 },
    { 0.534611048270758, 0.534611048270758, -0.069222096541517, 0.000927006328961 },
    
    { 0.202061394068290, 0.398969302965855, 0.398969302965855, 0.077149534914813 }, //
    { 0.398969302965855, 0.202061394068290, 0.398969302965855, 0.077149534914813 },
    { 0.398969302965855, 0.398969302965855, 0.202061394068290, 0.077149534914813 },
    
    { 0.593380199137435, 0.203309900431282, 0.203309900431282, 0.059322977380774 }, //
    { 0.203309900431282, 0.593380199137435, 0.203309900431282, 0.059322977380774 },
    { 0.203309900431282, 0.203309900431282, 0.593380199137435, 0.059322977380774 },
    
    
    { 0.761298175434837, 0.119350912282581, 0.119350912282581, 0.036184540503418 }, //
    { 0.119350912282581, 0.761298175434837, 0.119350912282581, 0.036184540503418 },
    { 0.119350912282581, 0.119350912282581, 0.761298175434837, 0.036184540503418 },
    
    { 0.935270103777448, 0.032364948111276, 0.032364948111276, 0.013659731002678 },
    { 0.032364948111276, 0.935270103777448, 0.032364948111276, 0.013659731002678 },
    { 0.032364948111276, 0.032364948111276, 0.935270103777448, 0.013659731002678 },
    
    { 0.050178138310495, 0.356620648261293, 0.593201213428213, 0.052337111962204 }, //
    { 0.050178138310495, 0.593201213428213, 0.356620648261293, 0.052337111962204 },
    { 0.356620648261293, 0.050178138310495, 0.593201213428213, 0.052337111962204 },
    { 0.356620648261293, 0.593201213428213, 0.050178138310495, 0.052337111962204 },
    { 0.593201213428213, 0.050178138310495, 0.356620648261293, 0.052337111962204 },
    { 0.593201213428213, 0.356620648261293, 0.050178138310495, 0.052337111962204 },
    
    { 0.021022016536166, 0.171488980304042, 0.807489003159792, 0.020707659639141 }, //
    { 0.021022016536166, 0.807489003159792, 0.171488980304042, 0.020707659639141 },
    { 0.171488980304042, 0.021022016536166, 0.807489003159792, 0.020707659639141 },
    { 0.171488980304042, 0.807489003159792, 0.021022016536166, 0.020707659639141 },
    { 0.807489003159792, 0.021022016536166, 0.171488980304042, 0.020707659639141 },
    { 0.807489003159792, 0.171488980304042, 0.021022016536166, 0.020707659639141 }
};

static double rule_12[][4] = {
    { 0.023565220452390, 0.488217389773805, 0.488217389773805, 0.025731066440455 }, //
    { 0.488217389773805, 0.023565220452390, 0.488217389773805, 0.025731066440455 },
    { 0.488217389773805, 0.488217389773805, 0.023565220452390, 0.025731066440455 },
    
    { 0.120551215411079, 0.439724392294460, 0.439724392294460, 0.043692544538038 }, //
    { 0.439724392294460, 0.120551215411079, 0.439724392294460, 0.043692544538038 },
    { 0.439724392294460, 0.439724392294460, 0.120551215411079, 0.043692544538038 },
    
    { 0.457579229975768, 0.271210385012116, 0.271210385012116, 0.062858224217885 }, //
    { 0.271210385012116, 0.457579229975768, 0.271210385012116, 0.062858224217885 },
    { 0.271210385012116, 0.271210385012116, 0.457579229975768, 0.062858224217885 },
    
    
    { 0.744847708916828, 0.127576145541586, 0.127576145541586, 0.034796112930709 }, //
    { 0.127576145541586, 0.744847708916828, 0.127576145541586, 0.034796112930709 },
    { 0.127576145541586, 0.127576145541586, 0.744847708916828, 0.034796112930709 },
    
    { 0.957365299093579, 0.021317350453210, 0.021317350453210, 0.006166261051559 },
    { 0.021317350453210, 0.957365299093579, 0.021317350453210, 0.006166261051559 },
    { 0.021317350453210, 0.021317350453210, 0.957365299093579, 0.006166261051559 },
    
    { 0.115343494534698, 0.275713269685514, 0.608943235779788, 0.040371557766381 }, //
    { 0.115343494534698, 0.608943235779788, 0.275713269685514, 0.040371557766381 },
    { 0.275713269685514, 0.115343494534698, 0.608943235779788, 0.040371557766381 },
    { 0.275713269685514, 0.608943235779788, 0.115343494534698, 0.040371557766381 },
    { 0.608943235779788, 0.115343494534698, 0.275713269685514, 0.040371557766381 },
    { 0.608943235779788, 0.275713269685514, 0.115343494534698, 0.040371557766381 },
    
    { 0.022838332222257, 0.281325580989940, 0.695836086787803, 0.022356773202303 }, //
    { 0.022838332222257, 0.695836086787803, 0.281325580989940, 0.022356773202303 },
    { 0.281325580989940, 0.022838332222257, 0.695836086787803, 0.022356773202303 },
    { 0.281325580989940, 0.695836086787803, 0.022838332222257, 0.022356773202303 },
    { 0.695836086787803, 0.022838332222257, 0.281325580989940, 0.022356773202303 },
    { 0.695836086787803, 0.281325580989940, 0.022838332222257, 0.022356773202303 },
    
    { 0.025734050548330, 0.116251915907597, 0.858014033544073, 0.017316231108659 }, //
    { 0.025734050548330, 0.858014033544073, 0.116251915907597, 0.017316231108659 },
    { 0.116251915907597, 0.025734050548330, 0.858014033544073, 0.017316231108659 },
    { 0.116251915907597, 0.858014033544073, 0.025734050548330, 0.017316231108659 },
    { 0.858014033544073, 0.025734050548330, 0.116251915907597, 0.017316231108659 },
    { 0.858014033544073, 0.116251915907597, 0.025734050548330, 0.017316231108659 }
};


static double rule_13[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.052520923400802 }, //
    
    { 0.009903630120591, 0.495048184939705, 0.495048184939705, 0.011280145209330 }, //
    { 0.495048184939705, 0.009903630120591, 0.495048184939705, 0.011280145209330 },
    { 0.495048184939705, 0.495048184939705, 0.009903630120591, 0.011280145209330 },
    
    { 0.062566729780852, 0.468716635109574, 0.468716635109574, 0.031423518362454 }, //
    { 0.468716635109574, 0.062566729780852, 0.468716635109574, 0.031423518362454 },
    { 0.468716635109574, 0.468716635109574, 0.062566729780852, 0.031423518362454 },
    
    
    { 0.170957326397447, 0.414521336801277, 0.414521336801277, 0.047072502504194 }, //
    { 0.414521336801277, 0.170957326397447, 0.414521336801277, 0.047072502504194 },
    { 0.414521336801277, 0.414521336801277, 0.170957326397447, 0.047072502504194 },
    
    { 0.541200855914337, 0.229399572042831, 0.229399572042831, 0.047363586536355 },
    { 0.229399572042831, 0.541200855914337, 0.229399572042831, 0.047363586536355 },
    { 0.229399572042831, 0.229399572042831, 0.541200855914337, 0.047363586536355 },
    
    { 0.771151009607340, 0.114424495196330, 0.114424495196330, 0.031167529045794 },
    { 0.114424495196330, 0.771151009607340, 0.114424495196330, 0.031167529045794 },
    { 0.114424495196330, 0.114424495196330, 0.771151009607340, 0.031167529045794 },
    
    { 0.950377217273082, 0.024811391363459, 0.024811391363459, 0.007975771465074 }, //
    { 0.024811391363459, 0.950377217273082, 0.024811391363459, 0.007975771465074 },
    { 0.024811391363459, 0.024811391363459, 0.950377217273082, 0.007975771465074 },
    
    { 0.094853828379579, 0.268794997058761, 0.636351174561660, 0.036848402728732 }, //
    { 0.094853828379579, 0.636351174561660, 0.268794997058761, 0.036848402728732 },
    { 0.268794997058761, 0.094853828379579, 0.636351174561660, 0.036848402728732 },
    { 0.268794997058761, 0.636351174561660, 0.094853828379579, 0.036848402728732 },
    { 0.636351174561660, 0.094853828379579, 0.268794997058761, 0.036848402728732 },
    { 0.636351174561660, 0.268794997058761, 0.094853828379579, 0.036848402728732 },
    
    { 0.018100773278807, 0.291730066734288, 0.690169159986905, 0.017401463303822 }, //
    { 0.018100773278807, 0.690169159986905, 0.291730066734288, 0.017401463303822 },
    { 0.291730066734288, 0.018100773278807, 0.690169159986905, 0.017401463303822 },
    { 0.291730066734288, 0.690169159986905, 0.018100773278807, 0.017401463303822 },
    { 0.690169159986905, 0.018100773278807, 0.291730066734288, 0.017401463303822 },
    { 0.690169159986905, 0.291730066734288, 0.018100773278807, 0.017401463303822 },
    
    { 0.022233076674090, 0.126357385491669, 0.851409537834241, 0.015521786839045 }, //
    { 0.022233076674090, 0.851409537834241, 0.126357385491669, 0.015521786839045 },
    { 0.126357385491669, 0.022233076674090, 0.851409537834241, 0.015521786839045 },
    { 0.126357385491669, 0.851409537834241, 0.022233076674090, 0.015521786839045 },
    { 0.851409537834241, 0.022233076674090, 0.126357385491669, 0.015521786839045 },
    { 0.851409537834241, 0.126357385491669, 0.022233076674090, 0.015521786839045 }
};


static double rule_14[][4] = {
    { 0.022072179275643, 0.488963910362179, 0.488963910362179, 0.021883581369429 }, //
    { 0.488963910362179, 0.022072179275643, 0.488963910362179, 0.021883581369429 },
    { 0.488963910362179, 0.488963910362179, 0.022072179275643, 0.021883581369429 },
    
    { 0.164710561319092, 0.417644719340454, 0.417644719340454, 0.032788353544125 }, //
    { 0.417644719340454, 0.164710561319092, 0.417644719340454, 0.032788353544125 },
    { 0.417644719340454, 0.417644719340454, 0.164710561319092, 0.032788353544125 },
    
    
    { 0.453044943382323, 0.273477528308839, 0.273477528308839, 0.051774104507292 }, //
    { 0.273477528308839, 0.453044943382323, 0.273477528308839, 0.051774104507292 },
    { 0.273477528308839, 0.273477528308839, 0.453044943382323, 0.051774104507292 },
    
    { 0.645588935174913, 0.177205532412543, 0.177205532412543, 0.042162588736993 },
    { 0.177205532412543, 0.645588935174913, 0.177205532412543, 0.042162588736993 },
    { 0.177205532412543, 0.177205532412543, 0.645588935174913, 0.042162588736993 },
    
    { 0.876400233818255, 0.061799883090873, 0.061799883090873, 0.014433699669777 },
    { 0.061799883090873, 0.876400233818255, 0.061799883090873, 0.014433699669777 },
    { 0.061799883090873, 0.061799883090873, 0.876400233818255, 0.014433699669777 },
    
    { 0.961218077502598, 0.019390961248701, 0.019390961248701, 0.004923403602400 }, //
    { 0.019390961248701, 0.961218077502598, 0.019390961248701, 0.004923403602400 },
    { 0.019390961248701, 0.019390961248701, 0.961218077502598, 0.004923403602400 },
    
    { 0.057124757403648, 0.172266687821356, 0.770608554774996, 0.024665753212564 }, //
    { 0.057124757403648, 0.770608554774996, 0.172266687821356, 0.024665753212564 },
    { 0.172266687821356, 0.057124757403648, 0.770608554774996, 0.024665753212564 },
    { 0.172266687821356, 0.770608554774996, 0.057124757403648, 0.024665753212564 },
    { 0.770608554774996, 0.057124757403648, 0.172266687821356, 0.024665753212564 },
    { 0.770608554774996, 0.172266687821356, 0.057124757403648, 0.024665753212564 },
    
    { 0.092916249356972, 0.336861459796345, 0.570222290846683, 0.038571510787061 }, //
    { 0.092916249356972, 0.570222290846683, 0.336861459796345, 0.038571510787061 },
    { 0.336861459796345, 0.092916249356972, 0.570222290846683, 0.038571510787061 },
    { 0.336861459796345, 0.570222290846683, 0.092916249356972, 0.038571510787061 },
    { 0.570222290846683, 0.092916249356972, 0.336861459796345, 0.038571510787061 },
    { 0.570222290846683, 0.336861459796345, 0.092916249356972, 0.038571510787061 },
    
    { 0.014646950055654, 0.298372882136258, 0.686980167808088, 0.014436308113534 }, //
    { 0.014646950055654, 0.686980167808088, 0.298372882136258, 0.014436308113534 },
    { 0.298372882136258, 0.014646950055654, 0.686980167808088, 0.014436308113534 },
    { 0.298372882136258, 0.686980167808088, 0.014646950055654, 0.014436308113534 },
    { 0.686980167808088, 0.014646950055654, 0.298372882136258, 0.014436308113534 },
    { 0.686980167808088, 0.298372882136258, 0.014646950055654, 0.014436308113534 },
    
    { 0.001268330932872, 0.118974497696957, 0.879757171370171, 0.005010228838501 }, //
    { 0.001268330932872, 0.879757171370171, 0.118974497696957, 0.005010228838501 },
    { 0.118974497696957, 0.001268330932872, 0.879757171370171, 0.005010228838501 },
    { 0.118974497696957, 0.879757171370171, 0.001268330932872, 0.005010228838501 },
    { 0.879757171370171, 0.001268330932872, 0.118974497696957, 0.005010228838501 },
    { 0.879757171370171, 0.118974497696957, 0.001268330932872, 0.005010228838501 }
    
};


struct rule {
	size_t num_points;
	double (*data)[4];
};

static struct rule rules[] = {
	{ sizeof(rule_1)/(sizeof(double)*4), rule_1 },
	{ sizeof(rule_2)/(sizeof(double)*4), rule_2 },
	{ sizeof(rule_3)/(sizeof(double)*4), rule_3 },
	{ sizeof(rule_4)/(sizeof(double)*4), rule_4 },
	{ sizeof(rule_5)/(sizeof(double)*4), rule_5 },
	{ sizeof(rule_6)/(sizeof(double)*4), rule_6 },
	{ sizeof(rule_7)/(sizeof(double)*4), rule_7 },
	{ sizeof(rule_8)/(sizeof(double)*4), rule_8 },
	{ sizeof(rule_9)/(sizeof(double)*4), rule_9 },
    { sizeof(rule_10)/(sizeof(double)*4), rule_10 },
    { sizeof(rule_11)/(sizeof(double)*4), rule_11 },
    { sizeof(rule_12)/(sizeof(double)*4), rule_12 },
    { sizeof(rule_13)/(sizeof(double)*4), rule_13 },
    { sizeof(rule_14)/(sizeof(double)*4), rule_14 }
};

} // namespace dunavant_quadratures
