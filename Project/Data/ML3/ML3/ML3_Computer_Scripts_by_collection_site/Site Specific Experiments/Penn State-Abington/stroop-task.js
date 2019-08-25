/* The script wrapper */
define(['app/API'], function(API) {

	API.addSettings('canvas',{
		maxWidth: 600,
		textSize: 9
	});
	// from where to take the images
	API.addSettings('base_url',{
		image : '/implicit/user/cebersole/manylabs3/ml3fullstudy/images/'
	});
	// Define where the logging records should be sent.
	API.addSettings('logger',{
		pulse: 21,
		url : '/implicit/PiPlayerApplet',
	});
	/***
	Stimulus sets
	***/
	//Colors
	API.addStimulusSets('red',[
		{media:'Red', css:{color:'red'}},
		{media:'Blue', css:{color:'red'}},
		{media:'Green', css:{color:'red'}}
	]);
	API.addStimulusSets('blue',[
		{media:'Red', css:{color:'blue'}},
		{media:'Blue', css:{color:'blue'}},
		{media:'Green', css:{color:'blue'}}
	]);
	API.addStimulusSets('green',[
		{media:'Red', css:{color:'green'}},
		{media:'Blue', css:{color:'green'}},
		{media:'Green', css:{color:'green'}}
	]);
	//Error and correct feedback.
	API.addStimulusSets('error',[
		{handle:'error',media:{image:'cross.png'}, location:{bottom:2}}
	]);
	API.addStimulusSets('correct',[
		{handle:'correct',media:{image:'check.png'}, location:{bottom:2}}
	]);
	//The basic trial
	API.addTrialSets('base',[{
		//Accepts three keys: 1, 2 and 3
		input: [
			{handle:'red',on:'keypressed',key:'1'},
			{handle:'blue',on:'keypressed',key:'2'},
			{handle:'green',on:'keypressed',key:'3'}
		],
		//What we present throughout the task (the key labels).
		layout: [
		//These are stimuli objects.
			{media:'1',location:{left:2,top:2},css:{background:'red',padding:'2%',fontSize:'1.5em'}},
			{media:'2',location:{top:2},css:{background:'blue',padding:'2%',fontSize:'1.5em'}},
			{media:'3',location:{right:2,top:2},css:{background:'green',padding:'2%',fontSize:'1.5em'}}
		],
		//All the events that can occur in a trial.
		interactions: [
			// Display the target stimulus.
			{
				conditions:[{type:'begin'}],
				actions: [{type:'showStim', handle: 'target'}]
			},
			// Correct response actions
			{
				conditions: [
				//The 'group' property in the trial's data should be one of the possible keys ('red', 'blue', 'green')
				//Here we test whether the response (that triggered the input) equals to the value of the group property of this trial.
					{type:'inputEqualsTrial',property:'group'}
				],
				//THe actions to perform when the response is correct.
				actions: [
					{type:'setTrialAttr', setter:{score:1}}, //Set the score to 1. That will be saved in trial_error in Project Implicit's server
					{type:'log'}, //Record this trial to send it to the server.
					{type:'showStim', handle: 'correct'}, //Show the correct feedback.
					{type:'removeInput',handle:['red','blue','green']}, //To stop accepting any responses, we remove the inputs.
					{type:'trigger', handle:'blank', duration:500} //In 500ms we will switch to the screen that is displayed between trials.
				]
			},
			// Incorrect response actions
			{
				conditions: [
					//The triggered input (by key response) does not equal the value of the 'group' variable in the trial's data.
					{type:'inputEqualsTrial',property:'group',negate:true},
					//The triggered input is one of the three key responses.
					{type:'inputEquals',value:['red','blue','green']}
				],
				actions: [
					{type:'setTrialAttr', setter:{score:0}}, //'score' must exist in the trial's data because Project Implicit's server expects score with each trial.
					{type:'log'}, //Log it (will be sent to the server).
					{type:'showStim', handle:'error'}, //show the error feedback.
					{type:'removeInput',handle:['red','blue','green']}, //Stop accepting any responses. We remove the inputs.
					{type:'trigger', handle:'blank', duration:500} //In 500ms we will switch to the screen that is displayed between trials.
				]
			},
			// Inter trial blank screen, shown for 500ms (the ITI)
			{
				conditions: [{type:'inputEquals', value:'blank'}],
				actions:[
					{type:'hideStim',handle:'All'}, //Hide all stimuli (but not the layout)
					{type:'removeInput',handle:['red','blue','green']}, ////To stop accepting any responses, we remove the inputs (probably already removed from the responses).
					{type:'trigger', handle:'end',duration:250} //In 500ms, we'll end this trial and the next trial will start.
				]
			},
			// End trial
			{
				conditions: [{type:'inputEquals', value:'end'}],
				actions:[
					{type:'endTrial'}
				]
			}
		]
	}]);
	//Inherit the 'base' trial with the three possible trials (one for each color) in this task.
	API.addTrialSets('red',[{
		inherit:'base', 
		data: {group:'red', condition:'red'}, //The condition will be saved in block_pairing_def. The group is the correct response.
		stimuli: [
			{inherit:{set:'red',type:'exRandom'}, handle:'target'}, //The target stimulus is from the 'red' stimulus set.
			{inherit:'correct'},
			{inherit:'error'}
		]
	}]);
	API.addTrialSets('blue',[{
		inherit:'base',
		data: {group:'blue', condition:'blue'},
		stimuli: [
			{inherit:{set:'blue',type:'exRandom'}, handle:'target'},
			{inherit:'correct'},
			{inherit:'error'}
		]
	}]);
	API.addTrialSets('green',[{
		inherit:'base',
		data: {group:'green', condition:'green'},
		stimuli: [
			{inherit:{set:'green',type:'exRandom'}, handle:'target'},
			{inherit:'correct'},
			{inherit:'error'}
		]
	}]);
	/**
	Define the sequence of trials
	**/
	API.addSequence([
		{
			mixer: 'random', //Randomize the 30 trials of this block.
			data: [
				{
					mixer: 'repeat',
					times: 21, //each type of trial will repeat 10 times in this block.
					data: [
						//Each trial inherits one of the three colors, and sets the block variable in the data.
						{inherit:'red', data:{block:1}}, 
						{inherit:'blue', data:{block:1}},
						{inherit:'green', data:{block:1}}
					]
				}
			]
		}
	]);

	// #### Activate the player
	API.play();
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//What to do at the end of the task.
	API.addSettings('hooks',{
		endTask: function(){
			$.post("/implicit/scorer", JSON.stringify({feedback:'This Stroop task has no feedback'})).always(function(){
				//Continue to the next task
				top.location.href = "/implicit/Study?tid="+xGetCookie("tid");
			});
		}
	});


});
/* don't forget to close the define wrapper */