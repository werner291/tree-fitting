extern crate piston_window;

use image::io::Reader as ImageReader;
use std::option::Option::Some;
use piston_window::{OpenGL, PistonWindow, WindowSettings, G2dTexture, Texture, TextureSettings, EventLoop, clear, image as pimage, line, Transformed, ellipse_from_to};
use image::{ImageBuffer, Rgba, RgbaImage, GenericImageView, Rgb, RgbImage, GrayImage};
use std::prelude::v1::Vec;
use std::convert::TryInto;
use nalgebra::{Point2, Vector2, Rotation2, Point3, Vector3};
use piston_window::math::Vec2d;
use piston_window::ellipse::circle;
use std::f64::consts::PI;
use std::iter::IntoIterator;
use std::collections::BinaryHeap;
use std::cmp::{Ord, Ordering, Eq};
use ndarray::{Array2, ArrayBase, OwnedRepr};
use image::imageops::blur;
use std::f32::INFINITY;
use ndarray_stats::QuantileExt;
use image::imageops::FilterType::CatmullRom;

struct Cursor {
    center: Point2<f64>,
    radius: f64,
    heading: Vector2<f64>,
}

impl Cursor {

    fn extremities(&self) -> (Point2<f64>, Point2<f64>) {

        let perpendicular : Vector2<f64> = &Rotation2::new(PI / 2.0) * &self.heading.normalize();

        (&self.center + self.radius * &perpendicular,
        &self.center - self.radius * &perpendicular)
    }


}

#[derive(PartialEq, PartialOrd)]
struct FrontierPoint {
    at: [usize;2],
    cost: f32
}

impl Eq for FrontierPoint {

}

impl Ord for FrontierPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.partial_cmp(&other.cost).unwrap()
    }
}

fn dijkstra_ish(from: [usize; 2], grid: &RgbaImage) -> Array2<f32> {

    //let grid = blur(grid, 5.0);

    let mut pqueue = BinaryHeap::new();

    let mut result = Array2::from_elem([grid.width() as usize, grid.height() as usize], INFINITY);
    result[from] = 0.0;

    pqueue.push(FrontierPoint {
        at:     from,
        cost: 0.0
    });

    let mut iii = 0;

    while let Some(FrontierPoint { at: [x,y], cost }) = pqueue.pop() {

        if iii % 100000 == 0 {
            println!("{} {}", pqueue.len(), result.iter().filter(|x| **x == INFINITY).count());
        }
        iii += 1;

        if cost <= result[[x,y]] {

            let neighbours = [
                [ x as i32 - 1, y as i32],
                [ x as i32 + 1, y as i32],
                [ x as i32, y as i32 - 1],
                [ x as i32, y as i32 + 1]];

            let [r, g, b, a] = grid.get_pixel(x as u32, y as u32).0;

            for [x2, y2] in neighbours.iter().cloned() {

                if x2 == -1 ||
                    y2 == -1 ||
                    x2 == grid.width() as i32 ||
                    y2 == grid.height() as i32 {
                    continue;
                }

                let [r2, g2, b2, a2] = grid.get_pixel(x2 as u32, y2 as u32).0;

                let distance = (r as f32 - r2 as f32).abs() +
                    (g as f32 - g2 as f32).abs() +
                    (b as f32 - b2 as f32).abs();

                if result[[x2 as usize, y2 as usize]] > result[[x, y]] + distance {
                    result[[x2 as usize, y2 as usize]] = result[[x, y]] + distance;
                    pqueue.push(FrontierPoint {
                        at: [x2 as usize, y2 as usize],
                        cost: result[[x, y]] + distance
                    })
                }
            }
        }
    }

    result
}

fn main() {
    let opengl = OpenGL::V3_2;

    let mut window: PistonWindow =
        WindowSettings::new("piston: image", [800, 267])
            .exit_on_esc(true)
            .graphics_api(opengl)
            .build()
            .unwrap();

    let img : RgbaImage = ImageReader::open("tree3.jpg").unwrap().decode().unwrap().to_rgba8();

    let distance = dijkstra_ish([225,225], &img);

    let furthest = *distance.max().unwrap();

    println!("{}", furthest);

    let distance_pixels = distance.t().iter().flat_map(|d| {
        let d_u8 = (d * 255.0 / furthest) as u8;
        vec![d_u8, d_u8, d_u8, 255]
    }).collect();

    let distance_gray = RgbaImage::from_raw(img.width(),img.height(), distance_pixels).unwrap();

    let tex: G2dTexture = Texture::from_image(
        &mut window.create_texture_context(),
        &img,
        &TextureSettings::new()
    ).unwrap();

    let gray_tex: G2dTexture = Texture::from_image(
        &mut window.create_texture_context(),
        &distance_gray,
        &TextureSettings::new()
    ).unwrap();

    window.set_lazy(true);


    // let cursor = Cursor {
    //     center: Point2::new(900.0,1000.0),
    //     radius: 100.0,
    //     heading: Vector2::new(0.0,-1.0)
    // };
    //

    let start = Point2::new(900.0,1000.0);
    let end = Point2::new(100.0, 275.0);

    while let Some(e) = window.next() {

        window.draw_2d(&e, |c, g, _| {

            let tf = c.transform;//.scale(0.25,0.25);

            clear([1.0; 4], g);
            pimage(&tex, tf, g);
            pimage(&gray_tex, tf.trans(img.width() as f64,0.0), g);

        });
    }
}

// fn mean_color_in_square(img: &RgbaImage, sample_middle: &Point2<f64>) -> Rgb<f32> {
//     let sample = img.view(sample_middle.x as u32 - 5, sample_middle.y as u32 - 5, 10, 10);
//
//     let mut sum = Vector3::new(0.0, 0.0, 0.0);
//
//     for (_,_,col) in sample.pixels() {
//         sum = sum + Vector3::new(col.0[0] as f32, col.0[1] as f32, col.0[2] as f32);
//     }
//
//     let mean = sum / (sample.width() * sample.height()) as f32;
//
//     Rgb([mean[0], mean[1], mean[2]])
// }