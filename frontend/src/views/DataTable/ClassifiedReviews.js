import React, { Component } from 'react';
import { Card, CardBody, Col, Row, Table } from 'reactstrap';

class DataTable extends Component {

    constructor(props){
        super(props)
        this.state = {
            // reviews: [
            //     {"review_id": 1, "sentence":"asasdasdad", "topic":"bedroom", "polarity": "0.23", "sentiment": "Positive"},
            //     {"review_id": 2, "sentence":"asasdasdad", "topic":"bedroom", "polarity": "0.23", "sentiment": "Positive"},
            //     {"review_id": 3, "sentence":"asasdasdad", "topic":"bedroom", "polarity": "0.23", "sentiment": "Positive"},
            //     {"review_id": 4, "sentence":"asasdasdad", "topic":"bedroom", "polarity": "0.23", "sentiment": "Positive"},
            //     {"review_id": 5, "sentence":"asasdasdad", "topic":"bedroom", "polarity": "0.23", 'sentiment': "Positive"}
            // ]
            title: props.title,
            header: props.header,
            reviews: props.reviews
        }
    }

    renderTableHeader() {
        let header = this.props.header
        return header.map((key, index) => {
            return <th key={index}>{key.toUpperCase()}</th>
        })
    }

    renderTableData() {
        return this.props.reviews.map((review) => {
            const {review_id, reviews, classification} = review
            return (
                <tr key={review_id}>
                    <td>{review_id}</td>
                    <td>{reviews}</td>
                    <td>{classification}</td>
                </tr>
            )
        })
    }
    
    render() {

        // need to make this component reusable
        return (
            <Row className="animated fadeIn">
                <Col>
                    <Card>
                        <CardBody>
                            <h1>{this.props.title}</h1>
                            <br></br>
                            <Table responsive striped hover className="table-outline mb-0 d-none d-sm-table">
                                <thead className="thead-light">
                                    <tr>{this.renderTableHeader()}</tr>
                                </thead>
                                <tbody>
                                    {this.renderTableData()}
                                </tbody>
                            </Table>
                        </CardBody>
                    </Card>
                </Col>
            </Row>
        )
    }
}

export default DataTable;