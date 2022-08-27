use std::{
    sync::mpsc::{channel, Receiver, Sender},
    thread::{available_parallelism, spawn, JoinHandle},
};

pub struct WorkerThread<Param, Result, Context> {
    input_sender: Sender<Option<(Context, Param)>>,
    output_receiver: Receiver<Result>,
    thread: JoinHandle<()>,
}

pub struct ForkJoinPool<Param, Result, Context> {
    threads: Vec<WorkerThread<Param, Result, Context>>,
}

impl<Param: 'static + Send, Result: 'static + Send, Context: 'static +  Send + Clone> ForkJoinPool<Param, Result, Context> {
    pub fn new(handler: fn(Context, Param) -> Result) -> ForkJoinPool<Param, Result, Context> {
        let num_threads = available_parallelism()
            .expect("Failed to find out available parallelism")
            .get();
        let mut threads = Vec::new();
        for _ in 0..num_threads {
            let (input_sender, input_receiver) = channel();
            let (output_sender, output_receiver) = channel();
            threads.push(WorkerThread {
                input_sender,
                output_receiver,
                thread: spawn(move || loop {
                    let input = input_receiver
                        .recv()
                        .expect("Failed to read from input receiver");
                    if let Some(input_value) = input {
                        output_sender
                            .send(handler(input_value.0, input_value.1))
                            .expect("Failed to send result");
                    } else {
                        break;
                    }
                }),
            });
        }
        ForkJoinPool { threads: threads }
    }
    pub fn exec_and_collect(
        &mut self,
        mut params: Vec<Param>,
        result_initializer: &dyn Fn() -> Result,
        collector: &dyn Fn(Result, Result) -> Result,
        context: Context,
    ) -> Result {
        let num_params = params.len();
        for i in 0..num_params {
            self.threads[i % self.threads.len()]
                .input_sender
                .send(Some((context.clone(), params.pop().unwrap())))
                .expect("Failed to send input");
        }
        let mut result = result_initializer();
        for i in 0..num_params {
            // Pass result second to have the results reversed because the code to send the params sends them backwards.
            result = collector(
                self.threads[i % self.threads.len()]
                    .output_receiver
                    .recv()
                    .expect("Failed to read from output sender"),
                result,
            );
        }
        result
    }
}
