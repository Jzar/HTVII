import React, { Component } from 'react';
import './App.css';
import TextFileReader from './File.js'
import axios from 'axios'


class App extends Component {

  constructor(props) {
    super(props);
    this.state = {
      value: '',
      show: false
      };

    this.handleChange = this.handleChange.bind(this);
    this.handleClick = this.handleClick.bind(this)
  }

  handleChange(event) {
    this.setState({value: event.target.value});
    console.log("change")
  }

  handleClick () {
    console.log('Click')
    var url = "http://www.sentiment140.com/api/classify?text=happy";
    axios.get(url)
    .then(response => console.log(response))
    }



  render() {
    var myTxt = require("./data/number.csv");

    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">Prometherus</h1>
        </header>
        <h3>
        Your ultimate Ethereum Predictor!
        </h3>
        <p>
        This is our estimated current Ethereum price in BTC
        <TextFileReader txt={myTxt}/>
        </p>
        <p>
        Here is a graph showing our prediction model vs the
        historical data on Ethereum price
        <br></br>
        <img src={require('./data/prediction_graph.png')} />
        </p>


      </div>
    );
  }
}

export default App;
