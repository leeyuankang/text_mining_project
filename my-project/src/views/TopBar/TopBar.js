import React, { Component } from 'react';
import { Card, CardBody, Col, Row } from 'reactstrap';

class TopBar extends Component {
    constructor(props){
        super(props)
        this.state = {
            no_of_reviews: props.no_of_reviews,
            sentiment_dist: props.sentiment_dist,
            no_of_topics: props.no_of_topics
        }
    }

    render() {

        return (
            <Row className="justify-content-md-center">
                
                <Col xs="12" sm="6" lg="3">
                    <Card className="text-white bg-yellow">
                        <CardBody className="pb-0">
                            <h2>No. Of Reviews Uploaded</h2>
                            <br></br>
                            <br></br>
                            <br></br>
                            <h2>{this.props.no_of_reviews}</h2>
                        </CardBody>
                    </Card>
                </Col>

                <Col xs="12" sm="6" lg="3">
                    <Card className="text-white bg-info">
                        <CardBody className="pb-0">
                            <h2>Sentiment Distribution</h2>
                            <br></br>
                            <div>
                                <h5>Positive: {this.props.senti_dist[0]}%</h5>
                                <h5>Neutral: {this.props.senti_dist[1]}%</h5>
                                <h5>Negative: {this.props.senti_dist[2]}%</h5>
                            </div>
                        </CardBody>
                    </Card>
                </Col>

                <Col xs="12" sm="6" lg="3">
                    <Card className="text-white bg-red">
                        <CardBody className="pb-0">
                            <h2>No. Of Topics Identified</h2>
                            <br></br>
                            <br></br>
                            <br></br>
                            <h2>{this.props.no_of_topics}</h2>
                        </CardBody>
                    </Card>
                </Col>

            </Row>
        )
    }
}

export default TopBar;