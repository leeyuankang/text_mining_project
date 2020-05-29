import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Nav, NavItem, NavLink } from 'reactstrap';

const propTypes = {
  children: PropTypes.node,
};

const defaultProps = {};

class DefaultSidebar extends Component {
  render() {

    // eslint-disable-next-line
    const { children, ...attributes } = this.props;

    return (
      <React.Fragment>
        <Nav vertical>
          <NavItem>
            <NavLink href="#">Link</NavLink>
          </NavItem>
          <NavItem>
            <NavLink href="#">Another Link</NavLink>
          </NavItem>
          <NavItem>
            <NavLink disabled href="#">Disabled Link</NavLink>
          </NavItem>
        </Nav>
      </React.Fragment>
    );
  }
}

DefaultSidebar.propTypes = propTypes;
DefaultSidebar.defaultProps = defaultProps;

export default DefaultSidebar;
