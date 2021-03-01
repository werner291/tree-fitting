use std::cmp::{Eq, Ord, Ordering};
use std::collections::BinaryHeap;
use std::convert::TryInto;
use std::f32::INFINITY;
use std::f64::consts::PI;
use std::iter::IntoIterator;
use std::option::Option::Some;
use std::prelude::v1::Vec;

use image::{GenericImageView, GrayImage, ImageBuffer, Rgb, Rgba, RgbaImage, RgbImage};
use image::imageops::blur;
use image::imageops::FilterType::CatmullRom;
use image::io::Reader as ImageReader;
use nalgebra::{Point2, Point3, Rotation2, Vector2, Vector3};
use ndarray::{Array2, ArrayBase, OwnedRepr};
use ndarray_stats::QuantileExt;
use piston_window::{clear, ellipse_from_to, EventLoop, G2dTexture, image as pimage, line, OpenGL, PistonWindow, Texture, TextureSettings, Transformed, WindowSettings};
use piston_window::ellipse::circle;
use piston_window::math::Vec2d;
use std::thread;
use std::sync::mpsc::{channel, TryRecvError};
use crate::dijkstra_method::{DijkstraApproach, StepResult};
use std::result::Result::{Ok, Err};
use std::time::{Instant, Duration};

mod dijkstra_method;

extern crate piston_window;

fn main() {
    let opengl = OpenGL::V3_2;

    let mut window: PistonWindow =
        WindowSettings::new("piston: image", [800, 267])
            .exit_on_esc(true)
            .graphics_api(opengl)
            .build()
            .unwrap();

    let mut texture_context = window.create_texture_context();

    let img : RgbaImage = ImageReader::open("tree3.jpg").unwrap().decode().unwrap().to_rgba8();

    let img_width = img.width();

    let tex: G2dTexture = Texture::from_image(
        &mut texture_context,
        &img,
        &TextureSettings::new()
    ).unwrap();

    let mut gray_tex = Texture::from_image(
        &mut texture_context,
        &img,
        &TextureSettings::new()
    ).unwrap();

    let (tx, rx) = channel();

    let join_handle = thread::spawn(move || {
        let mut algo = DijkstraApproach::new(&img, Point2::new(225,225));

        let mut since_last_send = Instant::now();

        loop {
            match algo.step() {

                StepResult::Done(result) => {
                    tx.send(result).unwrap();
                    break;
                }
                StepResult::NotDone(new_state) => {
                    if since_last_send.elapsed() > Duration::from_millis(100) {
                        since_last_send = Instant::now();
                        tx.send(new_state.state().to_owned()).unwrap();
                    }

                    algo = new_state;
                }
            }
        }
    });




    // window.set_lazy(true);

    while let Some(e) = window.next() {

        while let Ok(distance) = rx.try_recv() {
            let distance_gray = array2_to_image(distance);
            gray_tex.update(&mut texture_context, &distance_gray).unwrap();
            texture_context.encoder.flush(&mut window.device);
        }

        window.draw_2d(&e, |c, g, _| {

            let tf = c.transform;

            pimage(&tex, tf, g);

            pimage(&gray_tex, tf.trans(img_width as f64,0.0), g);

        });


    }

    join_handle.join().unwrap();
}

fn array2_to_image(distance: Array2<f32>) -> RgbaImage {

    let distance_without_inf = distance.mapv(|x| {
        if x == INFINITY {
            0.0
        } else {
            x
        }
    });

    let furthest = distance_without_inf.max().unwrap();

    let distance_pixels = distance.t().iter().flat_map(|d| {
        if *d == INFINITY {
            vec![255,0,255, 255]
        } else {
            let d_u8 = (d * 255.0 / furthest) as u8;
            vec![d_u8, d_u8, d_u8, 255]
        }
    }).collect();

    RgbaImage::from_raw(distance.nrows() as u32, distance.ncols() as u32, distance_pixels).unwrap()
}