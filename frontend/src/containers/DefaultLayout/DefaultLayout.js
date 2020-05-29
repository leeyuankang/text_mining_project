import React, { Component, Suspense } from 'react';
import { Container } from 'reactstrap';

import {
  AppHeader,
} from '@coreui/react';

const DefaultHeader = React.lazy(() => import('./DefaultHeader'));
const Chart = React.lazy(() => import('../../views/Charts/StackedBarCharts'))
const FileUpload = React.lazy(() => import('../../views/FileUpload/FileUpload'))
const DataTable = React.lazy(() => import('../../views/DataTable/DataTable'))
const TopBar = React.lazy(() => import("../../views/TopBar/TopBar"))
const ClassifiedReviews = React.lazy(() => import("../../views/DataTable/ClassifiedReviews"))

class DefaultLayout extends Component {

  loading = () => <div className="animated fadeIn pt-1 text-center">Loading...</div>

  constructor(props){
    super();
    this.state = {
      revLvlTitle: "Classified Reviews",
      revLvlHeader: ['review_id', 'review', 'classification'],
      revLvlData: [{"review_id": "empty", "reviews": "empty", "classification": "empty"}],
      no_of_reviews:0,
      senti_dist:[0,0,0],
      no_of_topics:0,
      chartData:{},
      senLvlTitle: "Detailed View of Sentiment Analysis (Sentence-Level)",
      senLvlHeader: ['review_id', 'sentence', 'topic', 'polarity', 'sentiment'],
      senLvlData: [{"review_id": "empty", "sentence": "empty", "topic": "empty", "polarity": "empty", "sentiment": "empty"}]
    }
  }

  onChangeLinkName(newName) {
    this.setState({
      homeLink: newName
    });
  }

  getRevLvlData(newRevData){
    this.setState({
      revLvlData: newRevData
    });
  }

  updateTopbar(newStats) {
    this.setState({
      no_of_reviews:newStats[0],
      senti_dist:newStats[1],
      no_of_topics:newStats[2]
    })
  }
  
  getChartData(newData){
    // perform Ajax calls here to retrieve data
    // maybe have to do callBack here
    this.setState({
      chartData:{
        labels:newData['topics'],
        // labels: Object.keys(newData),
        datasets: [
            {
                label:'Positive Sentiment',
                backgroundColor:"#77DD77",
                data:newData['positive'], // need to find a way to collate all positive percentage into one array
                order:3
            },
            {
              label:'Negative Sentiment',
              backgroundColor:"#FF6961",
              data:newData['negative'],
              order:1
            },
            {
              label:'Neutral Sentiment',
              backgroundColor:"#BABBBD", // need to think of new color
              data:newData['neutral'],
              order:2
            }
        ]
      }
    });
  }

  getSenLvlData(newSenData){
    this.setState({
      senLvlData: newSenData
    });
  }

  render() {
    return (
      <div className="app">
        <AppHeader fixed>
          <Suspense  fallback={this.loading()}>
            <DefaultHeader onLogout={e=>this.signOut(e)}/>
          </Suspense>
        </AppHeader>
        <div className="app-body">
          <main className="main">
            <Container fluid>
              <FileUpload
                changeClassRev={this.getRevLvlData.bind(this)}
                changeTopbarData={this.updateTopbar.bind(this)} 
                changeChartData={this.getChartData.bind(this)}
                changeSenLvlData={this.getSenLvlData.bind(this)}
              />
              <ClassifiedReviews 
                title={this.state.revLvlTitle}
                header={this.state.revLvlHeader}
                reviews={this.state.revLvlData}
              />
              <TopBar 
                no_of_reviews={this.state.no_of_reviews}
                senti_dist={this.state.senti_dist}
                no_of_topics={this.state.no_of_topics}
              />
              <Chart 
                homeLink={this.state.homeLink} 
                chartData={this.state.chartData} 
                location="New York" 
                legendPosition="bottom"
              />
              <DataTable 
                title={this.state.senLvlTitle}
                header={this.state.senLvlHeader}
                reviews={this.state.senLvlData}
              />
              {/* <DataTable reviews={this.state.reviews}/> */}
            </Container>
          </main>
        </div>
      </div>
    );
  }
}

export default DefaultLayout;
