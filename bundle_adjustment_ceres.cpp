#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

std::string file_name = "../dara/problem-16-22106-pre.txt";

void SolveBA(BALProblem &bal_problem);

int main (int argc, char** argv) {

    BALProblem bal_problem(file_name);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("../results/initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("../results/final.ply");

    return 0;
}