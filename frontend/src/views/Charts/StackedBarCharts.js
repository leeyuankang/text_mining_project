import React, { Component } from 'react';
import {Bar} from 'react-chartjs-2';
import {
    Card,
    CardBody,
    Row
} from 'reactstrap';

class Chart extends Component {

    constructor(props){
        super(props);
        this.state = {
            chartData: props.chartDatap
        }
    }

    static defaultProps = {
        displayTitle: true,
        displayLegend: true,
        legendPosition: 'right',
        location:'City'
    }

    render(){
        return (
            <div className="animated fadeIn">
                <Card>
                    <CardBody> 
                        <Row>
                            {/* If I want the data to be update-able */}
                            {/* <h1>{this.props.homeLink}</h1> */}
                            <Bar
                                data={this.props.chartData}
                                options={{
                                    title:{
                                        display:this.props.displayTitle,
                                        text:'Sentiment Analysis By Topics',
                                        fontSize:25
                                    },
                                    legend:{
                                        display:this.props.displayLegend,
                                        position:this.props.legendPosition
                                    },
                                    scales:{
                                        xAxes:[{
                                            stacked:true
                                        }],
                                        yAxes: [{
                                            stacked: true
                                        }]
                                    },
                                    // responsive:true, 
                                    // maintainAspectRatio:true,
                                    // tooltips:{
                                    //     mode:'index',
                                    //     intersect: false,
                                    //     callbacks:{
                                    //         label: function(tooltipItems, data) {
                                    //             return (parseFloat(tooltipItems.yLabel)).toFixed(1) + '%';
                                    //         }
                                    //     }
                                    // }
                                }}
                            />
                        </Row>
                    </CardBody>
                </Card>
                
            </div>
        )
    }
}

export default Chart;