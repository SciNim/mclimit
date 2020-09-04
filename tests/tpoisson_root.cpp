#include <iostream>
#include "TH1.h"
#include "TROOT.h"
#include "TSystem.h"
#include "mclimit.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TLimitDataSource.h"
#include "TConfidenceLevel.h"
#include "TVectorD.h"
#include "TObjString.h"

// csv parser
#include "csv.h"

using namespace ROOT;

double poisson(int k, double lam){
    return TMath::Power(lam, double (k)) / TMath::Factorial(k) * TMath::Exp(-lam);
}

int main(){

    TH1D sh = TH1D("", "", 15, 0.0, 14.0);
    TH1D bh = TH1D("", "", 15, 0.0, 14.0);
    TH1D dh = TH1D("", "", 15, 0.0, 14.0);
    TRandom3 rng = TRandom3(43);
    const int nmc = 100000;

    io::CSVReader<4> in("tpoisson.csv");
    in.read_header(io::ignore_extra_column, "k", "poisson", "back", "cand");
    int k; double pois, back, cand;
    while(in.read_row(k, pois, back, cand)){
	std::cout << k << " " << pois << " " << back << " " << cand << std::endl;
	sh.SetBinContent(k, pois);
	bh.SetBinContent(k, back);
	dh.SetBinContent(k, cand);
    }

    for(int i = 0; i < 15; i++){
	std::cout << bh.GetBinContent(i) << std::endl;
    }

    TLimitDataSource* dataSource = new TLimitDataSource();
    Double_t backEVal[2] = {0.1, 0.25};
    Double_t candEVal[2] = {0.1, 0.2};
    TVectorD backErr(2, backEVal);
    TVectorD candErr(2, candEVal);
    TObjArray names;
    TObjString n1("Tel");
    TObjString n2("Rad");
    names.AddLast(&n1);
    names.AddLast(&n2);

    dataSource->AddChannel(&sh, &bh, &dh, &candErr, &backErr, &names);
    TConfidenceLevel* limit = TLimit::ComputeLimit(dataSource, nmc, bool (0), &rng);
    std::cout << "  CLs    : " << limit->CLs()  << std::endl;
    std::cout << "  CLsb   : " << limit->CLsb() << std::endl;
    std::cout << "  CLb    : " << limit->CLb()  << std::endl;
    std::cout << "< CLs >  : " << limit->GetExpectedCLs_b()  << std::endl;
    std::cout << "< CLsb > : " << limit->GetExpectedCLsb_b() << std::endl;
    std::cout << "< CLb >  : " << limit->GetExpectedCLb_b()  << std::endl;

    //for(int i = 0; i < nmc; i++){
    //	std::cout << limit->fTSS[i] << std::endl;
    //}


    delete dataSource;
    delete limit;

    return 0;
}

void manual(){
    TH1D sh = TH1D("", "", 15, 0.0, 14.0);
    TH1D bh = TH1D("", "", 15, 0.0, 14.0);
    TH1D dh = TH1D("", "", 15, 0.0, 14.0);
    TRandom3 rng = TRandom3(43);

    for(int i = 0; i < 1000; i++){
	bh.Fill(rng.Poisson(7.0));
    }
    for(int i = 0; i < 100; i++){
	dh.Fill(rng.Poisson(7.0));
    }


    for(int i = 0; i < 15; i++){
	sh.SetBinContent(i, poisson(i, 5.0));
	std::cout << bh.GetBinContent(i) << " vs " << dh.GetBinContent(i) << " and " << sh.GetBinContent(i) << std::endl;
    }

    bh.Scale(1.0 / bh.GetEntries());
    dh.Scale(1.0 / dh.GetEntries());

    for(int i = 0; i < 15; i++){
	std::cout << bh.GetBinContent(i) << " vs " << dh.GetBinContent(i) << " and " << sh.GetBinContent(i) << std::endl;
    }



}
