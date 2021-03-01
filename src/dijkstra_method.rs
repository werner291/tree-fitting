//! Module containing a method based on Dijkstra's algorithm to discover tree structure in an image.
//!
//! It works as follows: you start with a picture of a tree (on the left), and pick a point somewhere
//! on it; just above the base of the trunk in my case. Then, run Dijkstra's algorithm, starting from
//! that point, using the difference in color between neighboring pixels as the edge weight. Finally,
//! translate the distance from the starting point to a gray-scale image (on the right).

use std::cmp::{Eq, Ord, Ordering};
use std::collections::BinaryHeap;
use std::f32::INFINITY;
use std::option::Option::Some;

use image::RgbaImage;
use ndarray::Array2;
use nalgebra::Point2;
use std::convert::Into;

/// A point on the exploration boundary.
///
/// Contains a point, and a cost.
///
/// `Ord` is implemented such that the point with lowest cost is maximum (highest priority)
#[derive(PartialEq, PartialOrd)]
struct FrontierPoint {
    at: [usize;2],
    cost: f32
}

impl Eq for FrontierPoint {

}

impl Ord for FrontierPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        // Cost interpreted as inverse priority.
        other.cost.partial_cmp(&self.cost).unwrap()
    }
}

/// Struct encapsulating the algorithm.
///
/// To run the algorithm:
///     - Construct a `DijkstraApproach` using `new`
///     - Call `step()` repeatedly.
///     - `step()` will eventually not return the result.
pub struct DijkstraApproach<'a> {
    state: Array2<f32>,
    queue: BinaryHeap<FrontierPoint>,
    image: &'a RgbaImage
}

pub enum StepResult<'a> {
    Done(Array2<f32>),
    NotDone(DijkstraApproach<'a>)
}

impl<'a> DijkstraApproach<'a> {

    /// Initialize the algorithm with an image and origin point.
    pub fn new(image: &'a RgbaImage, origin_point: Point2<u32>) -> Self {

        let mut queue = BinaryHeap::new();

        let mut state = Array2::from_elem([image.width() as usize, image.height() as usize], INFINITY);
        state[[origin_point.x as usize, origin_point.y as usize]] = 0.0;

        queue.push(FrontierPoint {
            at: [origin_point.x as usize, origin_point.y as usize],
            cost: 0.0
        });

        DijkstraApproach {
            state,
            queue,
            image
        }
    }

    /// Perform a single iteration of the algorithm.
    ///
    /// This method takes ownership of the state, and either returns the state after the iteration,
    /// or returns the result if the algorithm has terminated.
    pub fn step(mut self) -> StepResult<'a> {

        // Pop the current point with lowest cost (highest priority) from the queue.
        let FrontierPoint { at: [x,y], cost } = self.queue.pop()
            .expect("Algorithm should always have non-empty queue.");

        // Skip this point if a better cost has been found on a previous iteration.
        if cost <= self.state[[x as usize,y as usize]] {

            // Establish neighbour coordinates.
            let neighbours = [
                [ x as i32 - 1, y as i32],
                [ x as i32 + 1, y as i32],
                [ x as i32, y as i32 - 1],
                [ x as i32, y as i32 + 1]];

            // Copy the pixel from the original image.
            let [r, g, b, a] = self.image.get_pixel(x as u32, y as u32).0;

            // Explore all neighbours.
            for [x2, y2] in neighbours.iter().cloned() {

                // Skip the neighbour if it is out of bounds
                if x2 == -1 ||
                    y2 == -1 ||
                    x2 == self.image.width() as i32 ||
                    y2 == self.image.height() as i32 {
                    continue;
                }

                // Extract the pixel.
                let [r2, g2, b2, a2] = self.image.get_pixel(x2 as u32, y2 as u32).0;

                // Manhattan distance between color values is the edge traversal cost.
                let distance = (r as f32 - r2 as f32).abs() +
                    (g as f32 - g2 as f32).abs() +
                    (b as f32 - b2 as f32).abs();

                // Cost to visit neighbour if coming from current point.
                let candidate_cost = self.state[[x as usize, y as usize]] + distance;

                // If it's an improvement, store the new cost.
                if self.state[[x2 as usize, y2 as usize]] > candidate_cost {
                    self.state[[x2 as usize, y2 as usize]] = candidate_cost;
                    // Store as future point in the queue.
                    self.queue.push(FrontierPoint {
                        at: [x2  as usize, y2 as usize],
                        cost: self.state[[x, y]] + distance
                    })
                }
            }
        }

        // If done, return just the result, otherwise just return full state.
        if self.queue.is_empty() {
            StepResult::Done(self.state)
        } else {
            StepResult::NotDone(self)
        }
    }

    // Peek at the distance matrix.
    pub fn state(&self) -> &Array2<f32> {
        &self.state
    }
}
