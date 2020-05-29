import React, { Component } from "react";
import axios from 'axios';
import {
    Card,
    CardBody,
    Row
} from 'reactstrap';


class FileUpload extends Component {

    constructor(props){
        super(props);
        this.state = {
            selectedFile: null,
            homeLink: "Changed Link"
        }
    }




    fileSelectedHandler = event => {
        this.setState({
            selectedFile: event.target.files[0]
        })
    }

    fileUploadHandler = () => {
        const fd = new FormData();
        fd.append('file', this.state.selectedFile, this.state.selectedFile.name);
        axios.post('http://127.0.0.1:8001/reviews_analysis/file_upload', fd, {
            onUploadProgress: progressEvent => {
                console.log('Upload Progress: ' + Math.round(progressEvent.loaded / progressEvent.total * 100) + "%")
            }
        })
            .then(res => {
                // console.log(res['data']['chart_data']
                console.log(typeof(res['data']))
                this.props.changeClassRev(res['data']['classified_reviews'])
                this.props.changeTopbarData(res['data']['topbar_data'])
                this.props.changeChartData(res['data']['chart_data'])
                this.props.changeSenLvlData(res['data']['sen_lvl_data'])
                // this following line will help to change the state in DefaultLayout.js (Parent state)
            });
    }

    render() {
        return (
            <Card>
                <CardBody>
                    <Row>
                        <div className="fileUpload">
                            <h1>Upload Reviews CSV</h1>
                            <br></br>
                            <input 
                                type="file" 
                                onChange={this.fileSelectedHandler}
                            />
                            <button onClick={this.fileUploadHandler}>Upload</button>
                        </div>
                    </Row>
                </CardBody>
            </Card>
        
            
        )
    }
}

export default FileUpload;