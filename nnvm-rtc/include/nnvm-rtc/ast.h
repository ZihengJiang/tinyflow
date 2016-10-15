/*!
 *  Copyright (c) 2016 by Contributors
 * \file ast.h
 * \brief defines AST class for code generation
 */
#ifndef NNVM_RTC_AST_H_
#define NNVM_RTC_AST_H_

#include "./base.h"

namespace nnvm {
namespace rtc {

class AST {
 public:
  virtual ~AST() {}
  virtual std::string CodeGen() = 0;
};

class NumberAST : public AST {
 public:
  NumberAST(double val)
    : val_(val) {}
  inline std::string CodeGen() override {
    return std::to_string(val_);
  }
 private:
  double val_;
};

class VariableAST : public AST {
 public:
  VariableAST(std::string name)
    : name_(name) {}
  inline std::string CodeGen() override {
    // TODO(ziheng) Check whether variable exists?
    return name_;
  }
 private:
  std::string name_;
};

class BinaryAST : public AST {
 public:
  BinaryAST(char op, ASTPtr lhs, ASTPtr rhs)
    : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
  inline std::string CodeGen() override {
    return "(" + lhs_->CodeGen() + " " + op_ + " " + rhs_->CodeGen() + ")";
  }
 private:
  char op_;
  ASTPtr lhs_, rhs_;
};

class CallAST : public AST {
 public:
  CallAST(const std::string& callee, std::vector<ASTPtr> args)
    : callee_(callee), args_(std::move(args)) {}
  inline std::string CodeGen() override {
    std::string ret = callee_ + "(";
    for (uint32_t i = 0; i < args_.size(); ++i) {
      ret += std::move(args_[i]->CodeGen());
      ret += ", ";
    }
    ret.pop_back();
    ret.pop_back();
    ret += ")";
    return ret;
  }
 private:
  std::string callee_;
  std::vector<ASTPtr> args_;
};

class ArraySubscriptAST : public AST {
 public:
  ArraySubscriptAST(ASTPtr lhs, ASTPtr rhs)
    : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
  inline std::string CodeGen() override {
    return lhs_->CodeGen() + "[" + rhs_->CodeGen() + "]";
  }
 private:
  ASTPtr lhs_, rhs_;
};

inline ASTPtr operator+(ASTPtr lhs, ASTPtr rhs) {
  return ASTPtr(new BinaryAST('+', std::move(lhs), std::move(rhs)));
}

inline ASTPtr operator-(ASTPtr lhs, ASTPtr rhs) {
  return ASTPtr(new BinaryAST('-', std::move(lhs), std::move(rhs)));
}

inline ASTPtr operator*(ASTPtr lhs, ASTPtr rhs) {
  return ASTPtr(new BinaryAST('*', std::move(lhs), std::move(rhs)));
}

inline ASTPtr operator/(ASTPtr lhs, ASTPtr rhs) {
  return ASTPtr(new BinaryAST('/', std::move(lhs), std::move(rhs)));
}

}  // namespace rtc
}  // namespace nnvm

#endif  // NNVM_RTC_AST_H_
