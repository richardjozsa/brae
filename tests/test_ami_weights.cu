// Unit test for the cyclicAMI faceAreaWeightAMI geometric core (ami_detail polygon overlap / Sutherland-Hodgman).
// Validates the overlap-area computation that produces the AMI interpolation weights, on known cases.
#include "interface/ami_interface.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae::ami_detail;

static std::vector<vec2> sq(double x0, double y0, double x1, double y1) {
    return { {x0,y0}, {x1,y0}, {x1,y1}, {x0,y1} };   // CCW unit-ish rectangle
}

int main() {
    int fail = 0;
    auto chk = [&](const char* name, double got, double exp) {
        if (std::fabs(got - exp) > 1e-12) { std::printf("  %-22s FAIL got=%.15g exp=%.15g\n", name, got, exp); ++fail; }
    };
    chk("half-overlap",     overlapArea(sq(0,0,1,1), sq(0.5,0,1.5,1)), 0.5);
    chk("full-containment", overlapArea(sq(0,0,1,1), sq(-1,-1,2,2)),   1.0);
    chk("no-overlap",       overlapArea(sq(0,0,1,1), sq(2,2,3,3)),     0.0);
    chk("quarter-overlap",  overlapArea(sq(0,0,1,1), sq(0.5,0.5,1.5,1.5)), 0.25);
    chk("identical",        overlapArea(sq(0,0,1,1), sq(0,0,1,1)),     1.0);

    // conservation: a src face [0,2]x[0,1] tiled by 4 non-conforming tgt sub-faces -> full coverage (sum=srcArea)
    double tot = 0; for (int i = 0; i < 4; ++i) { double x0 = 0.5*i; tot += overlapArea(sq(0,0,2,1), sq(x0,0,x0+0.5,1)); }
    chk("conservation(4tiles)", tot, 2.0);

    // weights of a src face split across 2 equal tgt halves -> 0.5 each, sum 1
    double srcArea = 1.0;
    double w0 = overlapArea(sq(0,0,1,1), sq(0,0,0.5,1)) / srcArea;
    double w1 = overlapArea(sq(0,0,1,1), sq(0.5,0,1,1)) / srcArea;
    chk("weight-split-0", w0, 0.5); chk("weight-split-1", w1, 0.5); chk("weight-sum", w0+w1, 1.0);

    // winding independence: a clockwise target square must give the same overlap after orientCCW
    std::vector<vec2> cw = { {0.5,0}, {0.5,1}, {1.5,1}, {1.5,0} };  // CW
    orientCCW(cw);
    chk("cw-target",        overlapArea(sq(0,0,1,1), cw), 0.5);

    if (!fail) std::printf("test_ami_weights PASS (10 AMI overlap/weight checks)\n");
    return fail ? 1 : 0;
}
