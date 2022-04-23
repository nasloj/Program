import React from 'react';

import { BASE_URL } from '../../config';

import './SignUp.scss';

class SignUp extends React.Component {
  constructor() {
    super();
    this.state = {
      nameValue: '',
      emailValue: '',
      passwordValue: '',
      phoneNumValue: '',
    };
  }

  controlInput = event => {
    const { value, name } = event.target;
    console.log(this.state);
    if (name === 'inputName') {
      this.setState({
        nameValue: value,
      });
    } else if (name === 'inputPassword') {
      this.setState({
        passwordValue: value,
      });
    } else if (name === 'inputEmail') {
      this.setState({
        emailValue: value,
      });
    } else {
      this.setState({
        phoneNumValue: value,
      });
    }
  };

  // isAllValueValid = () => {
  //   const reg_pwd =
  //     /^[a-zA-Z0-9]+@[a-zA-Z0-9.]+\.[a-zA-Z0-9]+$/;
  //   const reg_phoneNum = /^[0-9]{3}[0-9]{3}[0-9]{4}$/;
  //   const reg_email = /^[a-zA-Z0-9]+@[a-zA-Z0-9.]+\.[a-zA-Z0-9]+$/;

  //   const isPasswordValid = reg_pwd.test(this.state.passwordValue);
  //   const isPhoneNumValid = reg_phoneNum.test(this.state.phoneNumValue);
  //   const isEmailValid = reg_email.test(this.state.emailValue);

  //   this.setState({
  //     isPasswordValid,
  //     isPhoneNumValid,
  //     isEmailValid,
  //   });
  // };

  goToMain = e => {
    // e.preventDefault();
    const { nameValue, emailValue, passwordValue, phoneNumValue } = this.state;

    // const isValidAll =
    //   nameValue && emailValue && passwordValue && phoneNumValue;

    // if (!isValidAll) return;

    fetch(`${BASE_URL}/users/signup`, {
      method: 'post',
      body: JSON.stringify({
        full_name: nameValue,
        email: emailValue,
        password: passwordValue,
        phone_number: phoneNumValue,
      }),
    })
      .then(response => response.json())
      .then(result => {
        if (result.message === 'SUCCESS') {
          alert('Thanks for signing up!');
          this.props.closeSignUp();
        } else {
          alert('Use already exsist!');
        }
      });
  };

  render() {
    const { isPasswordValid, isPhoneNumValid, isEmailValid, isNameValid } =
      this.state;
    const isAllValueTrue =
      isPasswordValid && isPhoneNumValid && isEmailValid && isNameValid;

    return (
      <div className="modalBg">
        <form className="signUpForm" onSubmit={this.goToMain}>
          <div className="signUpLogo">
            <img className="logoImage" alt="logo" src="images/mango.png" />
          </div>
          <div className="signUpTitle">Create Your Account</div>
          <div className="signUpContainer">
            <input
              type="text"
              name="inputName"
              className="inputName"
              placeholder="Name"
              onChange={this.controlInput}
              onKeyUp={this.isAllValueValid}
            />
            <input
              type="text"
              name="inputEmail"
              className="inputEmail"
              placeholder="Email Address"
              onChange={this.controlInput}
              onKeyUp={this.isAllValueValid}
            />
            <input
              type="text"
              name="inputPhoneNum"
              className="inputPhoneNum"
              placeholder="Phone Number"
              onChange={this.controlInput}
              onKeyUp={this.isAllValueValid}
            />
            <input
              type="password"
              name="inputPassword"
              className="inputPassword"
              placeholder="Password"
              onChange={this.controlInput}
              onKeyUp={this.isAllValueValid}
            />
            <button
              type="submit"
              className="submitInfoBtn"
              disabled={!isAllValueTrue}
            >
              Sign-up
            </button>
          </div>
          <div className="checkMembership">
            Already have an account? <a className="goToLogin">Sign-in</a>{' '}
          </div>
        </form>
      </div>
    );
  }
}

export default SignUp;