#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../src/functions.hpp"

class FunctionTests : public ::testing::Test {
protected:
    TFunctionFactory factory;
};

TEST_F(FunctionTests, CreateIdenticalFunction) {
    auto func = factory.Create("ident");
    ASSERT_EQ(func->ToString(), "x");
    EXPECT_NEAR((*func)(5), 5, 1e-8);
    EXPECT_NEAR(func->GetDerivative(5), 1, 1e-8);
}

TEST_F(FunctionTests, CreateConstantFunction) {
    auto func = factory.Create("const", 3.5);
    ASSERT_EQ(func->ToString(), "3.5");
    EXPECT_NEAR((*func)(5), 3.5, 1e-8);
    EXPECT_NEAR(func->GetDerivative(5), 0, 1e-8);
}

TEST_F(FunctionTests, CreatePowerFunction) {
    auto func = factory.Create("power", 2);
    ASSERT_EQ(func->ToString(), "x^2");
    EXPECT_NEAR((*func)(3), 9, 1e-8);
    EXPECT_NEAR(func->GetDerivative(3), 6, 1e-8);
}

TEST_F(FunctionTests, CreatePolynomialFunction) {
    auto func = factory.Create("polynomial", {1, -2, 1});  // x^2 - 2x + 1
    ASSERT_EQ(func->ToString(), "1 - 2*x + 1*x^2");
    EXPECT_NEAR((*func)(1), 0, 1e-8);
    EXPECT_NEAR(func->GetDerivative(1), 0, 1e-8);
}

TEST_F(FunctionTests, CreateExpFunction) {
    auto func = factory.Create("exp");
    ASSERT_EQ(func->ToString(), "e^x");
    EXPECT_NEAR((*func)(3), std::exp(3), 1e-8);
    EXPECT_NEAR(func->GetDerivative(3), std::exp(3), 1e-8);
}


TEST_F(FunctionTests, UnsupportedFunctionType) {
    EXPECT_THROW(factory.Create("unknown"), std::logic_error);
}

TEST_F(FunctionTests, NullPointerInOperation) {
    auto func = factory.Create("ident");
    EXPECT_THROW(auto result = func + nullptr, std::logic_error);
}

TEST_F(FunctionTests, UnsupportedFunctionInOperation) {
    class UnsupportedFunction : public TFunction {
    public:
        double operator()(double x) const override { return 0; }
        double GetDerivative(double x) const override { return 0; }
        std::string ToString() const override { return "Unsupported"; }
        std::string GetType() const override { return "unsupported"; }
    };
    auto func = factory.Create("ident");
    auto unsupported = std::make_shared<UnsupportedFunction>();
    EXPECT_THROW(auto result = func + unsupported, std::logic_error);
}


TEST_F(FunctionTests, CompositeFunctionOperations) {
    auto f = factory.Create("power", 2);  // f(x) = x^2
    auto g = factory.Create("const", 3); // g(x) = 3
    auto h = factory.Create("ident");    // h(x) = x
    auto p = factory.Create("polynomial", {1, -2, 1});  // p(x) = x^2 - 2x + 1

    // (f + g) -> x^2 + 3
    auto fg_sum = f + g;
    EXPECT_NEAR((*fg_sum)(2), 7, 1e-8);
    EXPECT_NEAR(fg_sum->GetDerivative(2), 4, 1e-8);

    // (f - h) -> x^2 - x
    auto fh_diff = f - h;
    EXPECT_NEAR((*fh_diff)(2), 2, 1e-8);
    EXPECT_NEAR(fh_diff->GetDerivative(2), 3, 1e-8);

    // (fg_sum * h) -> (x^2 + 3) * x
    auto product = fg_sum * h;
    EXPECT_NEAR((*product)(2), 14, 1e-8);
    EXPECT_NEAR(product->GetDerivative(2), 15, 1e-8);

    // (p / h) -> (x^2 - 2x + 1) / x
    auto division = p / h;
    EXPECT_NEAR((*division)(2), 0.5, 1e-8);
    EXPECT_NEAR(division->GetDerivative(2), 0.75, 1e-8);

    // ((f + g) * (f - h)) -> (x^2 + 3) * (x^2 - x)
    auto complex_composition = fg_sum * fh_diff;
    EXPECT_NEAR((*complex_composition)(2), 14, 1e-8);
    EXPECT_NEAR(complex_composition->GetDerivative(2), 29, 1e-8);
}


TEST_F(FunctionTests, SimpleQuadraticRoot) {
    // f(x) = x^2 - 4, roots: ±2
    auto func = factory.Create("polynomial", {-4, 0, 1});  // x^2 - 4
    double root = FindRoot(func, 10.0, 1000, 0.1);
    EXPECT_NEAR(root, 2.0, 1e-8);

    root = FindRoot(func, -10.0, 1000, 0.1);
    EXPECT_NEAR(root, -2.0, 1e-8);
}

TEST_F(FunctionTests, MultipleRoots) {
    // f(x) = x^3 - 6x^2 + 11x - 6, roots: 1, 2, 3
    auto func = factory.Create("polynomial", {-6, 11, -6, 1});  // x^3 - 6x^2 + 11x - 6

    // Check root finding with different initial guesses
    double root1 = FindRoot(func, 0.5, 1000, 0.1);
    EXPECT_NEAR(root1, 1.0, 1e-8);

    double root2 = FindRoot(func, 2.5, 1000, 0.1);
    EXPECT_NEAR(root2, 2.0, 1e-8);

    double root3 = FindRoot(func, 3.5, 1000, 0.1);
    EXPECT_NEAR(root3, 3.0, 1e-8);
}

TEST_F(FunctionTests, ExponentRoot) {
    // f(x) = e^x - 2, root: ln(2)
    auto func = factory.Create("exp");  // e^x
    auto shifted_func = func - factory.Create("const", 2);  // e^x - 2

    double root = FindRoot(shifted_func, 1.0, 1000, 0.1);
    EXPECT_NEAR(root, std::log(2), 1e-8);
}

TEST_F(FunctionTests, DerivativeTooSmall) {
    // f(x) = x^3, root: 0,  derivative is nearly zero near the root
    auto func = factory.Create("polynomial", {0, 0, 0, 1});  // x^3

    // Starting point close to the root, gradient descent should throw an exception
    EXPECT_THROW(FindRoot(func, 0.001, 1000, 0.1), std::logic_error);
}