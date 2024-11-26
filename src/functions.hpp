#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <functional>
#include <stdexcept>

class TFunction {
public:
    virtual ~TFunction() = default;

    virtual double operator()(double x) const = 0;

    virtual double GetDerivative(double x) const = 0;

    virtual std::string ToString() const = 0;

    virtual std::string GetType() const = 0;

    static const std::vector<std::string>& GetSupportedTypes() {
        static const std::vector<std::string> supportedTypes = {
            "ident", "const", "power", "exp", "polynomial", "composite"
        };
        return supportedTypes;
    }
};

using TFunctionPtr = std::shared_ptr<TFunction>;

class IdenticalFunction : public TFunction {
public:
    double operator()(double x) const override { return x; }

    double GetDerivative(double x) const override { return 1.0; }

    std::string ToString() const override { return "x"; }

    std::string GetType() const override { return "ident"; }
};

// Constant function f(x) = c
class ConstantFunction : public TFunction {
    double value;
public:
    explicit ConstantFunction(double value) : value(value) {}

    double operator()(double x) const override { return value; }

    double GetDerivative(double x) const override { return 0.0; }

    std::string ToString() const override {
        std::ostringstream oss;
        oss.precision(8);
        oss << std::noshowpoint << value;
        return oss.str();
    }

    std::string GetType() const override { return "const"; }
};

// Power function f(x) = x^n
class PowerFunction : public TFunction {
    double exponent;
public:
    explicit PowerFunction(double exponent) : exponent(exponent) {}

    double operator()(double x) const override { return std::pow(x, exponent); }

    double GetDerivative(double x) const override { return exponent * std::pow(x, exponent - 1); }

    std::string ToString() const override {
        std::ostringstream oss;
        oss << "x^" << exponent;
        return oss.str();
    }

    std::string GetType() const override { return "power"; }
};

// Exponential function f(x) = e^x
class ExpFunction : public TFunction {
public:
    double operator()(double x) const override { return std::exp(x); }

    double GetDerivative(double x) const override { return std::exp(x); }

    std::string ToString() const override { return "e^x"; }

    std::string GetType() const override { return "exp"; }
};

// Polynomial function f(x) = a_0 + a_1*x + a_2*x^2 + ...
class PolynomialFunction : public TFunction {
    std::vector<double> coefficients;
public:
    explicit PolynomialFunction(const std::vector<double>& coefficients) 
        : coefficients(coefficients) {}

    double operator()(double x) const override {
        double result = 0.0;
        double power = 1.0; // x^0
        for (double coeff : coefficients) {
            result += coeff * power;
            power *= x; // x^(i+1)
        }
        return result;
    }

    double GetDerivative(double x) const override {
        double result = 0.0;
        double power = 1.0; // x^0
        for (size_t i = 1; i < coefficients.size(); ++i) {
            result += i * coefficients[i] * power;
            power *= x; // x^(i-1)
        }
        return result;
    }

    std::string ToString() const override {
        std::ostringstream oss;
        oss.precision(8);
        oss << std::noshowpoint;
        bool first_term = true;

        for (size_t i = 0; i < coefficients.size(); ++i) {
            if (coefficients[i] == 0) continue;

            if (!first_term) {
                if (coefficients[i] > 0) {
                    oss << " + ";
                } else {
                    oss << " - ";
                }
            } else {
                if (coefficients[i] < 0) {
                    oss << "-";
                }
                first_term = false;
            }

            oss << std::abs(coefficients[i]);

            if (i > 0) oss << "*x";
            if (i > 1) oss << "^" << i;
        }

        return oss.str();
    }



    std::string GetType() const override { return "polynomial"; }
};


class TFunctionFactory {
public:
    TFunctionPtr Create(const std::string& type) {
        if (type == "ident") {
            return std::make_shared<IdenticalFunction>();
        } else if (type == "exp") {
            return std::make_shared<ExpFunction>();
        } else {
            throw std::logic_error("Unknown function type");
        }
    }

    TFunctionPtr Create(const std::string& type, double param) {
        if (type == "const") {
            return std::make_shared<ConstantFunction>(param);
        } else if (type == "power") {
            return std::make_shared<PowerFunction>(param);
        } else {
            throw std::logic_error("Unknown function type or invalid parameter count");
        }
    }

    TFunctionPtr Create(const std::string& type, const std::vector<double>& params) {
        if (type == "polynomial") {
            return std::make_shared<PolynomialFunction>(params);
        } else {
            throw std::logic_error("Unknown function type or invalid parameter count");
        }
    }
};


void ValidateFunctionTypes(const TFunctionPtr& left, const TFunctionPtr& right) {
    if (!left || !right) {
        throw std::logic_error("Null function pointer passed to operation");
    }

    const auto& supportedTypes = TFunction::GetSupportedTypes();
    if (std::find(supportedTypes.begin(), supportedTypes.end(), left->GetType()) == supportedTypes.end() ||
        std::find(supportedTypes.begin(), supportedTypes.end(), right->GetType()) == supportedTypes.end()) {
        throw std::logic_error("Unsupported function type in operation: " +
                               left->GetType() + " and " + right->GetType());
    }
}


class IBinaryOperation {
public:
    virtual ~IBinaryOperation() = default;

    virtual double Evaluate(double left, double right) const = 0;

    virtual double Derivative(double left, double right, double left_derivative, double right_derivative) const = 0;

    virtual std::string ToString() const = 0;
};

class AdditionOperation : public IBinaryOperation {
public:
    double Evaluate(double left, double right) const override {
        return left + right;
    }

    double Derivative(double left, double right, double left_derivative, double right_derivative) const override {
        return left_derivative + right_derivative;
    }

    std::string ToString() const override {
        return "+";
    }
};

class SubtractionOperation : public IBinaryOperation {
public:
    double Evaluate(double left, double right) const override {
        return left - right;
    }

    double Derivative(double left, double right, double left_derivative, double right_derivative) const override {
        return left_derivative - right_derivative;
    }

    std::string ToString() const override {
        return "-";
    }
};

class MultiplicationOperation : public IBinaryOperation {
public:
    double Evaluate(double left, double right) const override {
        return left * right;
    }

    double Derivative(double left, double right, double left_derivative, double right_derivative) const override {
        return left_derivative * right + left * right_derivative;
    }

    std::string ToString() const override {
        return "*";
    }
};

class DivisionOperation : public IBinaryOperation {
public:
    double Evaluate(double left, double right) const override {
        if (right == 0.0) {
            throw std::logic_error("Division by zero");
        }
        return left / right;
    }

    double Derivative(double left, double right, double left_derivative, double right_derivative) const override {
        if (right == 0.0) {
            throw std::logic_error("Division by zero in derivative");
        }
        return (left_derivative * right - left * right_derivative) / (right * right);
    }

    std::string ToString() const override {
        return "/";
    }
};


class CompositeFunction : public TFunction {
    TFunctionPtr left;
    TFunctionPtr right;
    std::shared_ptr<IBinaryOperation> operation;

public:
    CompositeFunction(TFunctionPtr left, TFunctionPtr right, std::shared_ptr<IBinaryOperation> operation)
        : left(left), right(right), operation(operation) {
        if (!left || !right || !operation) {
            throw std::logic_error("Invalid arguments for CompositeFunction");
        }
    }

    double operator()(double x) const override {
        return operation->Evaluate((*left)(x), (*right)(x));
    }

    double GetDerivative(double x) const override {
        return operation->Derivative((*left)(x), (*right)(x), left->GetDerivative(x), right->GetDerivative(x));
    }

    std::string ToString() const override {
        std::ostringstream oss;
        oss.precision(8);
        oss << std::noshowpoint;
        oss << "(" << left->ToString() << " " << operation->ToString() << " " << right->ToString() << ")";
        return oss.str();
    }

    std::string GetType() const override {
        return "composite";
    }
};


TFunctionPtr operator+(TFunctionPtr left, TFunctionPtr right) {
    ValidateFunctionTypes(left, right);
    return std::make_shared<CompositeFunction>(left, right, std::make_shared<AdditionOperation>());
}

TFunctionPtr operator-(TFunctionPtr left, TFunctionPtr right) {
    ValidateFunctionTypes(left, right);
    return std::make_shared<CompositeFunction>(left, right, std::make_shared<SubtractionOperation>());
}

TFunctionPtr operator*(TFunctionPtr left, TFunctionPtr right) {
    ValidateFunctionTypes(left, right);
    return std::make_shared<CompositeFunction>(left, right, std::make_shared<MultiplicationOperation>());
}

TFunctionPtr operator/(TFunctionPtr left, TFunctionPtr right) {
    ValidateFunctionTypes(left, right);
    return std::make_shared<CompositeFunction>(left, right, std::make_shared<DivisionOperation>());
}


double FindRoot(const TFunctionPtr& func, double initial_guess, size_t max_iterations, double learning_rate) {
    if (!func) {
        throw std::logic_error("Null function pointer passed to FindRoot");
    }

    double x = initial_guess;

    for (size_t i = 0; i < max_iterations; ++i) {
        double value = (*func)(x);
        double derivative = func->GetDerivative(x);

        // If the derivative is zero, avoid division by zero
        if (std::abs(derivative) < 1e-8) {
            throw std::logic_error("Derivative is too small, gradient descent cannot proceed");
        }

        x = x - learning_rate * value / derivative;
    }

    return x;
}